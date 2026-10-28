"""Command-Line Interface for CUDA_Translator."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

from cuda_translator.frontend.ptx_lexer import PTXLexer
from cuda_translator.frontend.ptx_parser import PTXParser
from cuda_translator.transform.warp_to_wavefront import WarpToWavefrontPass
from cuda_translator.transform.lds_normalizer import LDSNormalizerPass
from cuda_translator.transform.matrix_accelerator import MatrixAcceleratorPass
from cuda_translator.backend.vulkan_backend import VulkanBackend
from cuda_translator.backend.hip_backend import HIPBackend


console = Console()


def cmd_translate(args: argparse.Namespace) -> int:
    """Translate a PTX file to target Vulkan GLSL/SPIR-V or AMD HIP C++."""
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/] Input file '{input_path}' not found.")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        ptx_source = f.read()

    console.print(f"[bold cyan]Parsing PTX:[/] {input_path.name} ({len(ptx_source)} bytes)...")
    lexer = PTXLexer(ptx_source)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module(module_name=input_path.stem)

    console.print(f"[bold green]Ingested Kernels:[/] {[k.name for k in module.kernels]}")

    # Transformation Passes
    wavefront_size = 64 if args.target == "hip" else 32
    warp_pass = WarpToWavefrontPass(target_wavefront_size=wavefront_size)
    lds_pass = LDSNormalizerPass()
    matrix_pass = MatrixAcceleratorPass(target_backend=args.target)

    console.print("[dim]Running Warp-to-Wavefront Scheduling Pass...[/]")
    warp_pass.run_on_module(module)

    console.print("[dim]Running LDS & Cache Boundary Normalization Pass...[/]")
    lds_pass.run_on_module(module)

    console.print("[dim]Running Matrix Acceleration Routing Pass...[/]")
    matrix_pass.run_on_module(module)

    # Code Generation
    if args.target == "vulkan":
        backend = VulkanBackend()
        emitted_code = backend.generate(module)
    elif args.target == "hip":
        backend = HIPBackend(target_wavefront_size=wavefront_size)
        emitted_code = backend.generate(module)
    else:
        console.print(f"[bold red]Unsupported target:[/] {args.target}")
        return 1

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(emitted_code)
        console.print(f"[bold green]Successfully generated target code:[/] {out_path}")
    else:
        print(emitted_code)

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect and display IR details of a PTX kernel."""
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/] Input file '{input_path}' not found.")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        ptx_source = f.read()

    lexer = PTXLexer(ptx_source)
    tokens = lexer.tokenize()
    parser = PTXParser(tokens)
    module = parser.parse_module(module_name=input_path.stem)

    table = Table(title=f"CUDA_Translator IR Module: {module.name}")
    table.add_column("Kernel Name", style="cyan")
    table.add_column("Parameters", style="green")
    table.add_column("Basic Blocks", style="magenta")
    table.add_column("Instruction Count", style="yellow")
    table.add_column("Shared Memory", style="blue")

    for k in module.kernels:
        total_insts = sum(len(b.instructions) for b in k.blocks)
        params_str = ", ".join(str(p) for p in k.params)
        shared_str = ", ".join(f"{a.name}[{a.count}]" for a in k.shared_allocs) or "None"
        table.add_row(k.name, params_str, str(len(k.blocks)), str(total_insts), shared_str)

    console.print(table)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA_Translator: Cross-Architecture GPU Translation Layer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Translate subcommand
    translate_parser = subparsers.add_parser("translate", help="Translate PTX kernel to target architecture")
    translate_parser.add_argument("input", help="Path to input .ptx file")
    translate_parser.add_argument("--target", "-t", choices=["vulkan", "hip"], default="vulkan",
                                  help="Target architecture backend (vulkan, hip)")
    translate_parser.add_argument("--output", "-o", help="Output file path (default: stdout)")

    # Inspect subcommand
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Neutral IR from a PTX file")
    inspect_parser.add_argument("input", help="Path to input .ptx file")

    args = parser.parse_args()

    if args.command == "translate":
        sys.exit(cmd_translate(args))
    elif args.command == "inspect":
        sys.exit(cmd_inspect(args))


if __name__ == "__main__":
    main()
