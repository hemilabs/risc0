// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The bigint2 verify program: run by the executor and by the prover's preflight alike, so that a
//! guest whose witness the circuit would reject fails at execution — with the same memory writes,
//! in the same order, and the same cycle count as the proof will have.

use std::cmp::max;
use std::ops::{Add, AddAssign, Mul, MulAssign};

use anyhow::{anyhow, bail, ensure, Result};
use derive_more::Debug;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;
use risc0_binfmt::WordAddr;
use smallvec::{smallvec, SmallVec};

use super::{BigIntBytes, BigIntExec, BigIntWitness, BIGINT_WIDTH_BYTES, BIGINT_WIDTH_WORDS};
use crate::{
    execute::{
        r0vm::{LoadOp, Risc0Context},
        CycleState, WORD_SIZE,
    },
    MAX_INSN_CYCLES,
};

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "prove"), allow(dead_code))] // the preflight records every field
pub(crate) struct BigIntState {
    pub is_ecall: bool,
    pub mode: u32,
    pub pc: WordAddr,
    pub poly_op: PolyOp,
    pub coeff: u32,
    pub bytes: BigIntBytes,
    pub next_state: CycleState,
}

#[derive(Debug)]
#[debug("{poly_op:?}({mem_op:?}, c:{coeff}, r:{reg}, o:{offset})")]
pub(crate) struct Instruction {
    pub poly_op: PolyOp,
    pub mem_op: MemoryOp,
    pub coeff: i32,
    pub reg: u32,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum PolyOp {
    Reset,
    Shift,
    SetTerm,
    AddTotal,
    Carry1,
    Carry2,
    EqZero,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum MemoryOp {
    Read,
    Write,
    Check,
}

impl Instruction {
    // instruction encoding:
    // 3  2   2  2    1               0
    // 1  8   4  1    6               0
    // mmmmppppcccaaaaaoooooooooooooooo
    pub fn decode(insn: u32) -> Result<Self> {
        Ok(Self {
            mem_op: MemoryOp::from_u32((insn >> 28) & 0x0f)
                .ok_or_else(|| anyhow!("Invalid mem_op in bigint program"))?,
            poly_op: PolyOp::from_u32((insn >> 24) & 0x0f)
                .ok_or_else(|| anyhow!("Invalid poly_op in bigint program"))?,
            coeff: ((insn >> 21) & 0x07) as i32 - 4,
            reg: (insn >> 16) & 0x1f,
            offset: insn & 0xffff,
        })
    }
}

struct BigInt {
    state: BigIntState,
    program: BytePolyProgram,
}

/// Run the verify program of the bigint ecall `exec` decoded: from `exec.verify_program_ptr` to its
/// closing `Reset`, checking the witness as the circuit does and storing what its `Write`
/// instructions store, in their order. `on_cycle` sees each cycle (the preflight records them).
/// Returns the cycles it ran: the ecall's own and one per instruction.
///
/// A program that would take more cycles than one instruction may ([`MAX_INSN_CYCLES`]) is refused:
/// the executor splits segments assuming no instruction does, and one that did could overrun its
/// segment at some segment sizes and not at others.
pub(crate) fn run<C: Risc0Context>(
    ctx: &mut C,
    exec: &BigIntExec,
    mut on_cycle: impl FnMut(&mut C, CycleState, &BigIntState),
) -> Result<usize> {
    let mut bigint = BigInt {
        state: BigIntState {
            is_ecall: true,
            mode: exec.mode,
            pc: exec.verify_program_ptr,
            poly_op: PolyOp::Reset,
            coeff: 0,
            bytes: Default::default(),
            next_state: CycleState::BigIntStep,
        },
        program: BytePolyProgram::new(),
    };
    on_cycle(ctx, CycleState::BigIntEcall, &bigint.state);
    let mut cycles = 1;
    while bigint.state.next_state == CycleState::BigIntStep {
        cycles += 1;
        ensure!(
            cycles <= MAX_INSN_CYCLES,
            "bigint verify program runs over the {MAX_INSN_CYCLES} cycles one instruction may take"
        );
        bigint.step(ctx, &exec.witness)?;
        on_cycle(ctx, CycleState::BigIntStep, &bigint.state);
    }
    Ok(cycles)
}

impl BigInt {
    fn step(&mut self, ctx: &mut impl Risc0Context, witness: &BigIntWitness) -> Result<()> {
        self.state.pc.inc();
        let insn = Instruction::decode(ctx.load_u32(LoadOp::Record, self.state.pc)?)?;

        let base =
            ctx.load_aligned_addr_from_machine_register(LoadOp::Record, insn.reg as usize)?;
        let addr = base + insn.offset * BIGINT_WIDTH_WORDS as u32;

        tracing::trace!("step({:?}, {insn:?}, {addr:?})", self.state.pc);
        if insn.mem_op == MemoryOp::Check && insn.poly_op != PolyOp::Reset {
            if !self.program.in_carry {
                self.program.in_carry = true;
                self.program.total_carry = self.program.total.clone();
                let mut carry = 0;

                // Do carry propagation
                for i in 0..self.program.total_carry.len() {
                    let coeff = self.program.total_carry.get_mut(i);

                    *coeff += carry;
                    ensure!(*coeff % 256 == 0, "bad carry");
                    *coeff /= 256;
                    carry = *coeff;
                }
                tracing::trace!("carry propagate complete");
            }

            let base_point = 128 * 256 * 64;
            for (i, ret) in self.state.bytes.iter_mut().enumerate() {
                let offset = insn.offset as usize;
                let idx = offset * BIGINT_WIDTH_BYTES + i;
                // Past the polynomial's end the preflight's `get` panics: an error, here.
                ensure!(
                    idx < self.program.total_carry.len(),
                    "bigint check past the end of the polynomial"
                );
                let coeff = *self.program.total_carry.get(idx) as u32;
                let value = coeff.wrapping_add(base_point);
                match insn.poly_op {
                    PolyOp::Carry1 => *ret = ((value >> 14) & 0xff) as u8,
                    PolyOp::Carry2 => *ret = ((value >> 8) & 0x3f) as u8,
                    PolyOp::Shift | PolyOp::EqZero => *ret = (value & 0xff) as u8,
                    _ => {
                        bail!("Invalid poly_op in bigint program")
                    }
                }
            }
        } else if insn.mem_op == MemoryOp::Read {
            for i in 0..BIGINT_WIDTH_WORDS {
                let word = ctx.load_u32(LoadOp::Record, addr + i)?;
                for (j, byte) in word.to_le_bytes().iter().enumerate() {
                    self.state.bytes[i * WORD_SIZE + j] = *byte;
                }
            }
        } else if !addr.is_null() {
            self.state.bytes = *witness
                .get(&addr)
                .ok_or_else(|| anyhow!("Missing bigint witness: {addr:?}"))?;
            if insn.mem_op == MemoryOp::Write {
                let words: &[u32] = bytemuck::cast_slice(&self.state.bytes);
                for (i, word) in words.iter().enumerate() {
                    ctx.store_u32(addr + i, *word)?;
                }
            }
        }

        self.program.step(&insn, &self.state.bytes)?;

        self.state.is_ecall = false;
        self.state.poly_op = insn.poly_op;
        self.state.coeff = (insn.coeff + 4) as u32;
        self.state.next_state = if !self.state.is_ecall && insn.poly_op == PolyOp::Reset {
            CycleState::Decode
        } else {
            CycleState::BigIntStep
        };

        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct BytePolyProgram {
    pub(crate) in_carry: bool,
    pub(crate) poly: BytePolynomial,
    pub(crate) term: BytePolynomial,
    pub(crate) total: BytePolynomial,
    pub(crate) total_carry: BytePolynomial,
}

impl BytePolyProgram {
    pub(crate) fn new() -> Self {
        Self {
            in_carry: false,
            poly: BytePolynomial::zero(),
            term: BytePolynomial::one(),
            total: BytePolynomial::zero(),
            total_carry: BytePolynomial::default(),
        }
    }

    pub(crate) fn step(&mut self, insn: &Instruction, witness: &[u8]) -> Result<()> {
        let delta_poly = BytePolynomial {
            coeffs: SmallVec::from_iter(witness.chunks(BytePolyChunk::LANES as usize).map(|w| {
                let mut chunk = BytePolyChunk::ZERO;
                for (c, w) in chunk.as_array_mut().iter_mut().zip(w.iter()) {
                    *c = *w as i32;
                }
                chunk
            })),
        };

        let new_poly = &self.poly + &delta_poly;
        match insn.poly_op {
            PolyOp::Reset => {
                self.reset();
            }
            PolyOp::Shift => {
                self.poly = new_poly.shift();
            }
            PolyOp::SetTerm => {
                self.poly = BytePolynomial::zero();
                self.term = new_poly.clone();
            }
            PolyOp::AddTotal => {
                self.total += &new_poly * &self.term * insn.coeff;
                self.term = BytePolynomial::one();
                self.poly = BytePolynomial::zero();
            }
            PolyOp::Carry1 => {
                let mut coeffs = smallvec![];
                for _ in 0..(BIGINT_WIDTH_BYTES / BytePolyChunk::LANES as usize) {
                    coeffs.push(BytePolyChunk::splat(-128));
                }
                let neg_poly = BytePolynomial { coeffs };

                self.poly += (&delta_poly + neg_poly) * 64 * 256;
            }
            PolyOp::Carry2 => {
                self.poly += &delta_poly * 256;
            }
            PolyOp::EqZero => {
                let mut chunk = BytePolyChunk::ZERO;
                chunk.as_array_mut()[0] = -256;
                chunk.as_array_mut()[1] = 1;
                let bp = BytePolynomial {
                    coeffs: smallvec![chunk],
                };
                self.total += bp * &new_poly;
                self.total.eqz()?;
                self.reset();
                self.in_carry = false;
            }
        }

        tracing::trace!(
            "delta_poly[0]: {}, new_poly[0]: {}, poly[0]: {}, term[0]: {}, total[0]: {}",
            delta_poly.get(0),
            new_poly.get(0),
            self.poly.get(0),
            self.term.get(0),
            self.total.get(0),
        );

        Ok(())
    }

    fn reset(&mut self) {
        self.poly = BytePolynomial::zero();
        self.term = BytePolynomial::one();
        self.total = BytePolynomial::zero();
    }
}

type BytePolyChunk = wide::i32x4;

#[derive(Clone, Debug, Default)]
pub(crate) struct BytePolynomial {
    coeffs: SmallVec<[BytePolyChunk; 64 / BytePolyChunk::LANES as usize]>,
}

impl BytePolynomial {
    pub(crate) fn one() -> Self {
        let mut chunk = BytePolyChunk::ZERO;
        chunk.as_array_mut()[0] = 1;
        Self {
            coeffs: smallvec![chunk],
        }
    }

    pub(crate) fn zero() -> Self {
        Self {
            coeffs: smallvec![BytePolyChunk::ZERO],
        }
    }

    pub(crate) fn shift(&self) -> Self {
        let mut ret = self.clone();
        for _ in 0..(BIGINT_WIDTH_BYTES / BytePolyChunk::LANES as usize) {
            ret.coeffs.insert(0, BytePolyChunk::ZERO);
        }
        ret
    }

    pub(crate) fn eqz(&self) -> Result<()> {
        ensure!(
            self.coeffs
                .iter()
                .all(|&coeff| coeff == BytePolyChunk::ZERO),
            "Invalid eqz in bigint program"
        );
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.coeffs.len() * BytePolyChunk::LANES as usize
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &i32 {
        assert!(idx < self.len());
        unsafe { self.get_unchecked(idx) }
    }

    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, idx: usize) -> &i32 {
        let i = idx / BytePolyChunk::LANES as usize;
        let j = idx % BytePolyChunk::LANES as usize;
        self.coeffs.get_unchecked(i).as_array_ref().get_unchecked(j)
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, idx: usize) -> &mut i32 {
        assert!(idx < self.len());
        unsafe { self.get_unchecked_mut(idx) }
    }

    #[inline(always)]
    pub(crate) unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut i32 {
        let i = idx / BytePolyChunk::LANES as usize;
        let j = idx % BytePolyChunk::LANES as usize;
        self.coeffs
            .get_unchecked_mut(i)
            .as_array_mut()
            .get_unchecked_mut(j)
    }
}

#[inline(always)]
fn byte_poly_add(lhs: &BytePolynomial, rhs: &BytePolynomial) -> BytePolynomial {
    let mut ret = smallvec![BytePolyChunk::ZERO; max(lhs.coeffs.len(), rhs.coeffs.len())];
    for (i, coeff) in ret.iter_mut().enumerate() {
        if i < lhs.coeffs.len() {
            *coeff += lhs.coeffs[i];
        }
        if i < rhs.coeffs.len() {
            *coeff += rhs.coeffs[i];
        }
    }
    BytePolynomial { coeffs: ret }
}

#[inline(always)]
fn byte_poly_add_assign(lhs: &mut BytePolynomial, rhs: &BytePolynomial) {
    lhs.coeffs
        .resize(max(lhs.coeffs.len(), rhs.coeffs.len()), BytePolyChunk::ZERO);

    for (coeff, &rhs) in lhs.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
        *coeff += rhs;
    }
}

#[inline(always)]
fn byte_poly_mul(lhs: &BytePolynomial, rhs: &BytePolynomial) -> BytePolynomial {
    let mut ret = BytePolynomial {
        coeffs: smallvec![BytePolyChunk::ZERO; lhs.coeffs.len() + rhs.coeffs.len()],
    };

    for i in 0..lhs.len() {
        let lhs_chunk = BytePolyChunk::splat(*lhs.get(i));

        if i % BytePolyChunk::LANES as usize == 0 {
            let start = i / BytePolyChunk::LANES as usize;
            for (ret_chunk, rhs_chunk) in ret.coeffs[start..].iter_mut().zip(rhs.coeffs.iter()) {
                *ret_chunk += *rhs_chunk * lhs_chunk;
            }
        } else {
            for (j, rhs_chunk) in rhs.coeffs.iter().enumerate() {
                let new_chunk = *rhs_chunk * lhs_chunk;
                for (k, v) in new_chunk.as_array_ref().iter().enumerate() {
                    unsafe {
                        *ret.get_unchecked_mut(i + j * BytePolyChunk::LANES as usize + k) += v
                    };
                }
            }
        }
    }
    ret
}

#[inline(always)]
fn byte_poly_mul_const(lhs: &BytePolynomial, rhs: i32) -> BytePolynomial {
    let mut ret = lhs.coeffs.clone();
    for coeff in ret.iter_mut() {
        *coeff *= BytePolyChunk::splat(rhs);
    }
    BytePolynomial { coeffs: ret }
}

#[inline(always)]
fn byte_poly_mul_assign_const(lhs: &mut BytePolynomial, rhs: i32) {
    for coeff in &mut lhs.coeffs {
        *coeff *= BytePolyChunk::splat(rhs);
    }
}

impl Add<&BytePolynomial> for &BytePolynomial {
    type Output = BytePolynomial;

    #[inline(always)]
    fn add(self, rhs: &BytePolynomial) -> Self::Output {
        byte_poly_add(self, rhs)
    }
}

impl Add<BytePolynomial> for &BytePolynomial {
    type Output = BytePolynomial;

    #[inline(always)]
    fn add(self, rhs: BytePolynomial) -> Self::Output {
        byte_poly_add(self, &rhs)
    }
}

impl AddAssign<&BytePolynomial> for BytePolynomial {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &BytePolynomial) {
        byte_poly_add_assign(self, rhs)
    }
}

impl AddAssign<BytePolynomial> for BytePolynomial {
    #[inline(always)]
    fn add_assign(&mut self, rhs: BytePolynomial) {
        byte_poly_add_assign(self, &rhs)
    }
}

impl Mul<&BytePolynomial> for &BytePolynomial {
    type Output = BytePolynomial;

    #[inline(always)]
    fn mul(self, rhs: &BytePolynomial) -> Self::Output {
        byte_poly_mul(self, rhs)
    }
}

impl Mul<&BytePolynomial> for BytePolynomial {
    type Output = BytePolynomial;

    #[inline(always)]
    fn mul(self, rhs: &BytePolynomial) -> Self::Output {
        byte_poly_mul(&self, rhs)
    }
}

impl Mul<i32> for &BytePolynomial {
    type Output = BytePolynomial;

    #[inline(always)]
    fn mul(self, rhs: i32) -> Self::Output {
        byte_poly_mul_const(self, rhs)
    }
}

impl Mul<i32> for BytePolynomial {
    type Output = BytePolynomial;

    #[inline(always)]
    fn mul(self, rhs: i32) -> Self::Output {
        byte_poly_mul_const(&self, rhs)
    }
}

impl MulAssign<i32> for BytePolynomial {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: i32) {
        byte_poly_mul_assign_const(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use risc0_binfmt::ByteAddr;

    use super::*;
    use crate::execute::{platform::*, r0vm::TestRisc0Context};

    /// Register holding the arena the test programs address.
    const ARENA: usize = REG_A0;
    const BASE: ByteAddr = ByteAddr(ZERO_PAGE_END_ADDR.0 + 0x1000);
    const PROGRAM: ByteAddr = ByteAddr(ZERO_PAGE_END_ADDR.0 + 0x8000);

    fn insn(mem_op: MemoryOp, poly_op: PolyOp, offset: u32) -> u32 {
        ((mem_op as u32) << 28)
            | ((poly_op as u32) << 24)
            | (4 << 21)
            | ((ARENA as u32) << 16)
            | offset
    }

    /// A context with the arena register set and `program` laid out after the verify pointer (the
    /// first instruction is read at the pointer plus one), and the ecall's decoded `exec`.
    fn setup(program: &[u32], witness: BigIntWitness) -> (TestRisc0Context, BigIntExec) {
        let mut ctx = TestRisc0Context::default();
        ctx.store_register(MACHINE_REGS_ADDR.waddr(), ARENA, BASE.0)
            .unwrap();
        let ptr = PROGRAM.waddr();
        for (i, word) in program.iter().enumerate() {
            ctx.store_u32(ptr + 1 + i, *word).unwrap();
        }
        let exec = BigIntExec {
            mode: 0,
            verify_program_ptr: ptr,
            witness,
        };
        (ctx, exec)
    }

    fn witness(entries: &[(u32, BigIntBytes)]) -> BigIntWitness {
        entries
            .iter()
            .map(|(offset, bytes)| (BASE.waddr() + offset * BIGINT_WIDTH_WORDS as u32, *bytes))
            .collect()
    }

    fn run_counting(ctx: &mut TestRisc0Context, exec: &BigIntExec) -> (Result<usize>, usize) {
        let mut seen = 0;
        let r = run(ctx, exec, |_, _, _| seen += 1);
        (r, seen)
    }

    #[test]
    fn writes_what_its_write_instructions_name_and_counts_its_cycles() {
        let bytes: BigIntBytes = std::array::from_fn(|i| i as u8 + 1);
        // Offset 1 is in the witness but no instruction writes it.
        let (mut ctx, exec) = setup(
            &[insn(MemoryOp::Write, PolyOp::Reset, 0)],
            witness(&[(0, bytes), (1, [9; BIGINT_WIDTH_BYTES])]),
        );
        let (r, seen) = run_counting(&mut ctx, &exec);
        assert_eq!(r.unwrap(), 2, "the ecall's cycle and one per instruction");
        assert_eq!(seen, 2, "the preflight is told of each of them");
        let written = ctx
            .load_region(LoadOp::Peek, BASE, BIGINT_WIDTH_BYTES)
            .unwrap();
        assert_eq!(written, bytes);
        ctx.assert_uninit(BASE + BIGINT_WIDTH_BYTES);
    }

    #[test]
    fn a_read_sees_memory_before_a_later_write() {
        // Read offset 0 (memory as it is), then write the witness there: in program order.
        let old: BigIntBytes = [7; BIGINT_WIDTH_BYTES];
        let new: BigIntBytes = [0; BIGINT_WIDTH_BYTES];
        let (mut ctx, exec) = setup(
            &[
                insn(MemoryOp::Read, PolyOp::SetTerm, 0),
                insn(MemoryOp::Write, PolyOp::Reset, 0),
            ],
            witness(&[(0, new)]),
        );
        ctx.store_region(BASE, &old).unwrap();
        assert_eq!(run(&mut ctx, &exec, |_, _, _| {}).unwrap(), 3);
        assert_eq!(
            ctx.load_region(LoadOp::Peek, BASE, BIGINT_WIDTH_BYTES)
                .unwrap(),
            new
        );
    }

    #[test]
    fn a_write_with_no_witness_is_refused() {
        let (mut ctx, exec) = setup(
            &[insn(MemoryOp::Write, PolyOp::Reset, 1)],
            witness(&[(0, [1; BIGINT_WIDTH_BYTES])]),
        );
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Missing bigint witness"), "{err}");
    }

    #[test]
    fn a_witness_that_fails_its_check_is_refused() {
        // EqZero over a nonzero witness: the circuit's "Invalid eqz".
        let (mut ctx, exec) = setup(
            &[insn(MemoryOp::Write, PolyOp::EqZero, 0)],
            witness(&[(0, [1; BIGINT_WIDTH_BYTES])]),
        );
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Invalid eqz"), "{err}");

        // A zero one passes, and the program goes on to its Reset.
        let (mut ctx, exec) = setup(
            &[
                insn(MemoryOp::Write, PolyOp::EqZero, 0),
                insn(MemoryOp::Write, PolyOp::Reset, 0),
            ],
            witness(&[(0, [0; BIGINT_WIDTH_BYTES])]),
        );
        assert_eq!(run(&mut ctx, &exec, |_, _, _| {}).unwrap(), 3);
    }

    #[test]
    fn an_undecodable_instruction_is_refused() {
        let (mut ctx, exec) = setup(&[0x0f00_0000], BigIntWitness::new());
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Invalid poly_op"), "{err}");
    }

    #[test]
    fn a_program_longer_than_an_instruction_may_take_is_refused() {
        // Never a Reset: it would run on through memory. Stopped at the per-instruction budget.
        let program = vec![insn(MemoryOp::Read, PolyOp::SetTerm, 0); MAX_INSN_CYCLES];
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let (r, seen) = run_counting(&mut ctx, &exec);
        assert!(r.unwrap_err().to_string().contains("runs over"));
        assert_eq!(
            seen, MAX_INSN_CYCLES,
            "it runs the budget, and not one cycle more"
        );

        // One shorter, ending in its Reset, fits exactly.
        let mut program = vec![insn(MemoryOp::Read, PolyOp::SetTerm, 0); MAX_INSN_CYCLES - 2];
        program.push(insn(MemoryOp::Read, PolyOp::Reset, 0));
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        assert_eq!(run(&mut ctx, &exec, |_, _, _| {}).unwrap(), MAX_INSN_CYCLES);
    }
}
