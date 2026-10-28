"""PTX Parser: Ingests PTX tokens and builds the Neutral GPU Intermediate Representation."""

from __future__ import annotations
import re
from typing import List, Optional, Dict, Any
from cuda_translator.frontend.ptx_lexer import Token, TokenType, PTXLexer
from cuda_translator.ir.types import (
    BaseType, AddressSpace, parse_type, B32, U32, U64, F32, PRED, PointerType
)
from cuda_translator.ir.instructions import (
    Instruction, AllocaInst, LoadInst, StoreInst, BinaryOpInst, UnaryOpInst,
    FusedMultiplyAddInst, CompareInst, ConvertInst, BranchInst, BarrierInst,
    SubgroupShuffleInst, SubgroupBallotInst, CooperativeMatrixLoadInst,
    CooperativeMatrixStoreInst, CooperativeMatrixMulAddInst, ReturnInst
)
from cuda_translator.ir.module import KernelParam, BasicBlock, Kernel, Module


class PTXParserError(Exception):
    """Exception raised for syntax or semantic errors in PTX parsing."""
    pass


class PTXParser:
    """Parses PTX assembly into a Neutral GPU IR Module."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]

    def peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def match(self, expected_type: TokenType, expected_val: Optional[str] = None) -> bool:
        tok = self.current()
        if tok.type != expected_type:
            return False
        if expected_val is not None and tok.value != expected_val:
            return False
        return True

    def consume(self, expected_type: Optional[TokenType] = None, expected_val: Optional[str] = None) -> Token:
        tok = self.current()
        if expected_type is not None and tok.type != expected_type:
            raise PTXParserError(f"Expected token type {expected_type.name}, got {tok.type.name} ('{tok.value}') at L{tok.line}:{tok.column}")
        if expected_val is not None and tok.value != expected_val:
            raise PTXParserError(f"Expected value '{expected_val}', got '{tok.value}' at L{tok.line}:{tok.column}")
        self.pos += 1
        return tok

    def parse_module(self, module_name: str = "cuda_module") -> Module:
        """Parse full PTX translation unit."""
        module = Module(name=module_name)

        while not self.match(TokenType.EOF):
            tok = self.current()

            # Global directives: .version, .target, .address_size, etc.
            if tok.type == TokenType.DIRECTIVE:
                if tok.value in {".version", ".target", ".address_size"}:
                    self.consume()
                    while not self.match(TokenType.SEMICOLON) and not self.match(TokenType.EOF):
                        # Some versions don't end in semicolon
                        if self.current().line != tok.line:
                            break
                        self.consume()
                    if self.match(TokenType.SEMICOLON):
                        self.consume()
                elif tok.value in {".visible", ".entry"}:
                    kernel = self.parse_kernel()
                    module.add_kernel(kernel)
                else:
                    # Skip other global declarations for now
                    self.consume()
            else:
                self.consume()

        return module

    def parse_kernel(self) -> Kernel:
        """Parse a kernel function (.entry)."""
        if self.match(TokenType.DIRECTIVE, ".visible"):
            self.consume()

        self.consume(TokenType.DIRECTIVE, ".entry")
        name_tok = self.consume(TokenType.IDENTIFIER)
        kernel = Kernel(name=name_tok.value)

        # Parse parameter list in parentheses
        if self.match(TokenType.LPAREN):
            self.consume(TokenType.LPAREN)
            while not self.match(TokenType.RPAREN) and not self.match(TokenType.EOF):
                if self.match(TokenType.DIRECTIVE, ".param"):
                    self.consume(TokenType.DIRECTIVE, ".param")
                    type_tok = self.consume(TokenType.DIRECTIVE)
                    p_type = parse_type(type_tok.value)
                    # Parameter name
                    param_name = self.consume(TokenType.IDENTIFIER).value
                    # Check if pointer (often 64-bit address)
                    is_ptr = p_type.bitwidth == 64
                    kernel.params.append(KernelParam(name=param_name, data_type=p_type, is_pointer=is_ptr))

                if self.match(TokenType.COMMA):
                    self.consume(TokenType.COMMA)
                elif not self.match(TokenType.RPAREN):
                    self.consume()

            self.consume(TokenType.RPAREN)

        # Parse Kernel Body { ... }
        self.consume(TokenType.LBRACE)
        current_block = kernel.get_or_create_block("entry")

        while not self.match(TokenType.RBRACE) and not self.match(TokenType.EOF):
            tok = self.current()

            # Check for label (e.g. $L__BB0_1:)
            if (tok.type == TokenType.IDENTIFIER or tok.type == TokenType.NUMBER) and self.peek().type == TokenType.COLON:
                lbl = tok.value
                self.consume()  # identifier
                self.consume(TokenType.COLON)
                current_block = kernel.get_or_create_block(lbl)
                continue

            # Check for register declarations: .reg .b32 %r<10>; or .reg .pred %p0;
            if tok.type == TokenType.DIRECTIVE and tok.value == ".reg":
                self.consume(TokenType.DIRECTIVE, ".reg")
                reg_type = parse_type(self.consume(TokenType.DIRECTIVE).value)
                while not self.match(TokenType.SEMICOLON) and not self.match(TokenType.EOF):
                    reg_tok = self.consume()
                    reg_val = reg_tok.value
                    # Range like %r<10>
                    m = re.match(r"(%[a-zA-Z0-9_]+)<([0-9]+)>", reg_val)
                    if m:
                        base = m.group(1)
                        count = int(m.group(2))
                        for i in range(count):
                            kernel.registers[f"{base}{i}"] = reg_type
                    elif reg_val.startswith("%"):
                        kernel.registers[reg_val] = reg_type
                    if self.match(TokenType.COMMA):
                        self.consume(TokenType.COMMA)
                self.consume(TokenType.SEMICOLON)
                continue

            # Check for shared memory allocation: .shared .align 4 .b8 smem[1024];
            if tok.type == TokenType.DIRECTIVE and tok.value == ".shared":
                self.consume(TokenType.DIRECTIVE, ".shared")
                # Parse optional alignment
                align = 4
                if self.match(TokenType.DIRECTIVE, ".align"):
                    self.consume()
                    align = int(self.consume(TokenType.NUMBER).value)
                elem_type = parse_type(self.consume(TokenType.DIRECTIVE).value)
                name = self.consume(TokenType.IDENTIFIER).value
                count = 1
                if self.match(TokenType.LBRACKET):
                    self.consume(TokenType.LBRACKET)
                    count = int(self.consume(TokenType.NUMBER).value)
                    self.consume(TokenType.RBRACKET)
                self.consume(TokenType.SEMICOLON)
                kernel.shared_allocs.append(AllocaInst(
                    name=name, element_type=elem_type, count=count, space=AddressSpace.SHARED
                ))
                continue

            # Parse Instruction
            inst = self.parse_instruction()
            if inst:
                current_block.add_instruction(inst)

        self.consume(TokenType.RBRACE)
        return kernel

    def parse_instruction(self) -> Optional[Instruction]:
        """Parse a single PTX instruction statement."""
        predicate: Optional[str] = None
        pred_negated = False

        # Handle predicate guard: @%p1 or @!%p1
        if self.match(TokenType.PREDICATE):
            pred_tok = self.consume(TokenType.PREDICATE)
            val = pred_tok.value
            if val.startswith("@!"):
                pred_negated = True
                predicate = val[2:]
            else:
                predicate = val[1:]

        tok = self.current()
        if tok.type not in {TokenType.IDENTIFIER, TokenType.DIRECTIVE}:
            self.consume()
            return None

        mnemonic = tok.value
        self.consume()

        # Collect operands until semicolon
        operands: List[str] = []
        while not self.match(TokenType.SEMICOLON) and not self.match(TokenType.EOF):
            if self.match(TokenType.COMMA):
                self.consume(TokenType.COMMA)
                continue
            if self.match(TokenType.LBRACKET):
                # Memory operand: [%rd1] or [%rd1 + 8]
                self.consume(TokenType.LBRACKET)
                mem_expr = self.consume().value
                if self.match(TokenType.PLUS):
                    self.consume(TokenType.PLUS)
                    offset = self.consume().value
                    mem_expr += f" + {offset}"
                elif self.match(TokenType.MINUS):
                    self.consume(TokenType.MINUS)
                    offset = self.consume().value
                    mem_expr += f" - {offset}"
                self.consume(TokenType.RBRACKET)
                operands.append(f"[{mem_expr}]")
            else:
                operands.append(self.consume().value)

        if self.match(TokenType.SEMICOLON):
            self.consume(TokenType.SEMICOLON)

        parts = mnemonic.split(".")
        opcode = parts[0]
        dtype = parse_type(parts[-1]) if len(parts) > 1 else B32

        # Memory Load
        if opcode == "ld":
            space = AddressSpace.GLOBAL
            if "shared" in parts:
                space = AddressSpace.SHARED
            elif "param" in parts:
                space = AddressSpace.PARAM
            elif "local" in parts:
                space = AddressSpace.LOCAL

            dest = operands[0] if len(operands) > 0 else ""
            ptr_raw = operands[1] if len(operands) > 1 else ""
            ptr, offset = self._extract_ptr_offset(ptr_raw)
            return LoadInst(predicate=predicate, pred_negated=pred_negated,
                            dest=dest, ptr=ptr, data_type=dtype, space=space, offset=offset)

        # Memory Store
        elif opcode == "st":
            space = AddressSpace.GLOBAL
            if "shared" in parts:
                space = AddressSpace.SHARED
            elif "local" in parts:
                space = AddressSpace.LOCAL

            ptr_raw = operands[0] if len(operands) > 0 else ""
            src = operands[1] if len(operands) > 1 else ""
            ptr, offset = self._extract_ptr_offset(ptr_raw)
            return StoreInst(predicate=predicate, pred_negated=pred_negated,
                             ptr=ptr, src=src, data_type=dtype, space=space, offset=offset)

        # Binary & Unary Arithmetic
        elif opcode in {"add", "sub", "mul", "div", "rem", "and", "or", "xor", "shl", "shr"}:
            if len(operands) >= 3:
                return BinaryOpInst(predicate=predicate, pred_negated=pred_negated,
                                    dest=operands[0], op=opcode, src1=operands[1], src2=operands[2], data_type=dtype)
            elif len(operands) == 2:
                return UnaryOpInst(predicate=predicate, pred_negated=pred_negated,
                                   dest=operands[0], op=opcode, src=operands[1], data_type=dtype)

        # Fused Multiply Add / MAD
        elif opcode in {"fma", "mad"}:
            if len(operands) >= 4:
                return FusedMultiplyAddInst(predicate=predicate, pred_negated=pred_negated,
                                            dest=operands[0], src_a=operands[1], src_b=operands[2], src_c=operands[3], data_type=dtype)

        # Comparison / Set Predicate (setp)
        elif opcode == "setp":
            cond = parts[1] if len(parts) > 1 else "eq"
            if len(operands) >= 3:
                return CompareInst(predicate=predicate, pred_negated=pred_negated,
                                   dest=operands[0], cond=cond, src1=operands[1], src2=operands[2], data_type=dtype)

        # Branch
        elif opcode == "bra":
            target = operands[0] if len(operands) > 0 else ""
            return BranchInst(predicate=predicate, pred_negated=pred_negated, target=target, condition=predicate)

        # Synchronization Barrier
        elif opcode == "bar":
            return BarrierInst(predicate=predicate, pred_negated=pred_negated)

        # Register Shuffle (Warp Intrinsic)
        elif opcode == "shfl":
            mode = "bfly" if "bfly" in parts else "down" if "down" in parts else "up" if "up" in parts else "idx"
            dest = operands[0] if len(operands) > 0 else ""
            src = operands[1] if len(operands) > 1 else ""
            delta = operands[2] if len(operands) > 2 else "1"
            return SubgroupShuffleInst(predicate=predicate, pred_negated=pred_negated,
                                       dest=dest, src=src, delta_or_lane=delta, mode=mode)

        # Warp Ballot (vote.sync.ballot)
        elif opcode == "vote":
            dest = operands[0] if len(operands) > 0 else ""
            pred_src = operands[1] if len(operands) > 1 else ""
            return SubgroupBallotInst(predicate=predicate, pred_negated=pred_negated,
                                      dest=dest, pred=pred_src, width=32)

        # Cooperative Matrix / WMMA
        elif opcode == "wmma":
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "load":
                layout = "col_major" if "col" in parts else "row_major"
                return CooperativeMatrixLoadInst(
                    dest_frag=operands[0] if len(operands) > 0 else "",
                    ptr=operands[1] if len(operands) > 1 else "",
                    stride=operands[2] if len(operands) > 2 else "16",
                    layout=layout
                )
            elif sub == "store":
                layout = "col_major" if "col" in parts else "row_major"
                return CooperativeMatrixStoreInst(
                    ptr=operands[0] if len(operands) > 0 else "",
                    src_frag=operands[1] if len(operands) > 1 else "",
                    stride=operands[2] if len(operands) > 2 else "16",
                    layout=layout
                )
            elif sub == "mma":
                return CooperativeMatrixMulAddInst(
                    dest_frag=operands[0] if len(operands) > 0 else "",
                    frag_a=operands[1] if len(operands) > 1 else "",
                    frag_b=operands[2] if len(operands) > 2 else "",
                    frag_c=operands[3] if len(operands) > 3 else operands[0]
                )

        # Register Move / Conversion
        elif opcode in {"mov", "cvt", "cvta"}:
            dest = operands[0] if len(operands) > 0 else ""
            src = operands[1] if len(operands) > 1 else ""
            return ConvertInst(predicate=predicate, pred_negated=pred_negated,
                               dest=dest, src=src, from_type=dtype, to_type=dtype)

        # Return
        elif opcode == "ret":
            return ReturnInst(predicate=predicate, pred_negated=pred_negated)

        # Default fallback: Treat as convert/assign
        dest = operands[0] if len(operands) > 0 else ""
        src = operands[1] if len(operands) > 1 else ""
        return ConvertInst(predicate=predicate, pred_negated=pred_negated,
                           dest=dest, src=src, from_type=dtype, to_type=dtype)

    def _extract_ptr_offset(self, mem_str: str) -> tuple[str, int]:
        """Extract base pointer and byte offset from [%rd1 + 16]."""
        cleaned = mem_str.strip("[] \t")
        if "+" in cleaned:
            parts = cleaned.split("+")
            return parts[0].strip(), int(parts[1].strip(), 0)
        elif "-" in cleaned:
            parts = cleaned.split("-")
            return parts[0].strip(), -int(parts[1].strip(), 0)
        return cleaned, 0
