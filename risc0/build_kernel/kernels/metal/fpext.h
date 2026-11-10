// Copyright 2024 RISC Zero, Inc.
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

#pragma once

/// \file
/// Defines FpExt, a finite field F_p^4, based on Fp via the irreducible polynomial x^4 - 11.

#include <metal_stdlib>

#include "fp.h"

// Defines instead of constexpr to appease CUDAs limitations around constants.
// undef'd at the end of this file.
#define BETA Fp(11)
#define NBETA Fp(Fp::P - 11)

/// Instances of FpExt are element of a finite field F_p^4.  They are represented as elements of
/// F_p[X] / (X^4 - 11). Basically, this is a 'big' finite field (about 2^128 elements), which is
/// used when the security of various operations depends on the size of the field.  It has the field
/// Fp as a subfield, which means operations by the two are compatible, which is important.  The
/// irreducible polynomial was chosen to be the simplest possible one, x^4 - B, where 11 is the
/// smallest B which makes the polynomial irreducible.
struct FpExt {
  /// The elements of FpExt, elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*x^4
  Fp elems[4];

  /// Default constructor makes the zero elements
  constexpr FpExt() {}

  /// Address-space qualified constructors, so that `threadgroup FpExt y[N]`, `device` and
  /// program-scope `constant` objects can be declared (see fp.h). The `threadgroup` and `constant`
  /// ones build `elems` in the member initializer from the raw words (a `constant` object can't be
  /// assigned to, and a thread `Fp` can't copy-construct an `Fp` in another address space); the
  /// `device` ones assign in the body (metal 32023.922 rejects the array member initializer for
  /// a `device` object: "no viable constructor copying array element").
  constexpr FpExt() threadgroup {}
  constexpr FpExt() device {}
  constexpr FpExt() constant {}

  /// Initialize from uint32_t
  explicit constexpr FpExt(uint32_t x) {
    elems[0] = x;
    elems[1] = 0;
    elems[2] = 0;
    elems[3] = 0;
  }

  explicit constexpr FpExt(uint32_t x) threadgroup : elems{x, 0u, 0u, 0u} {}
  explicit constexpr FpExt(uint32_t x) device {
    elems[0] = x;
  }
  explicit constexpr FpExt(uint32_t x) constant : elems{x, 0u, 0u, 0u} {}

  /// Convert from Fp to FpExt.
  explicit constexpr FpExt(Fp x) {
    elems[0] = x;
    elems[1] = 0;
    elems[2] = 0;
    elems[3] = 0;
  }

  explicit constexpr FpExt(Fp x) threadgroup : elems{{x.asRaw(), true}, 0u, 0u, 0u} {}
  explicit constexpr FpExt(Fp x) device {
    elems[0] = x;
  }
  explicit constexpr FpExt(Fp x) constant : elems{{x.asRaw(), true}, 0u, 0u, 0u} {}

  /// Explicitly construct an FpExt from parts
  constexpr FpExt(Fp a, Fp b, Fp c, Fp d) {
    elems[0] = a;
    elems[1] = b;
    elems[2] = c;
    elems[3] = d;
  }

  constexpr FpExt(Fp a, Fp b, Fp c, Fp d) threadgroup
      : elems{{a.asRaw(), true}, {b.asRaw(), true}, {c.asRaw(), true}, {d.asRaw(), true}} {}
  constexpr FpExt(Fp a, Fp b, Fp c, Fp d) device {
    elems[0] = a;
    elems[1] = b;
    elems[2] = c;
    elems[3] = d;
  }
  constexpr FpExt(Fp a, Fp b, Fp c, Fp d) constant
      : elems{{a.asRaw(), true}, {b.asRaw(), true}, {c.asRaw(), true}, {d.asRaw(), true}} {}

  constexpr inline FpExt zeroize() {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i].zeroize();
    }
    return *this;
  }

  // Implement the addition/subtraction overloads. Compound assignment modifies the object in
  // place, so it is a member, one overload per address space the object can live in.
  constexpr FpExt operator+=(FpExt rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] += rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator+=(FpExt rhs) device {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] += rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator+=(FpExt rhs) threadgroup {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] += rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator-=(FpExt rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] -= rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator-=(FpExt rhs) device {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] -= rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator-=(FpExt rhs) threadgroup {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] -= rhs.elems[i];
    }
    return *this;
  }

  // Binary, unary and comparison operators are by-value hidden friends (free functions found by
  // argument-dependent lookup), so either operand may live in any address space (thread,
  // threadgroup, device, constant): it is copied into a thread value first.
  friend constexpr FpExt operator+(FpExt lhs, FpExt rhs) {
    lhs += rhs;
    return lhs;
  }

  friend constexpr FpExt operator-(FpExt lhs, FpExt rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend constexpr FpExt operator-(FpExt x) { return FpExt() - x; }

  // Implement the simple multiplication case by the subfield Fp
  constexpr FpExt operator*=(Fp rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] *= rhs;
    }
    return *this;
  }

  constexpr FpExt operator*=(Fp rhs) device {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] *= rhs;
    }
    return *this;
  }

  constexpr FpExt operator*=(Fp rhs) threadgroup {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] *= rhs;
    }
    return *this;
  }

  friend constexpr FpExt operator*(FpExt lhs, Fp rhs) {
    lhs *= rhs;
    return lhs;
  }

  /// Overload for case where LHS is Fp
  friend constexpr FpExt operator*(Fp lhs, FpExt rhs) { return rhs * lhs; }

  // Now we get to the interesting case of multiplication.  Basically, multiply out the polynomial
  // representations, and then reduce module x^4 - B, which means powers >= 4 get shifted back 4 and
  // multiplied by -beta.  We could write this as a double loops with some if's and hope it gets
  // unrolled properly, but it's small enough to just hand write.
  friend constexpr FpExt operator*(FpExt lhs, FpExt rhs) {
    // Rename the element arrays to something small for readability
#define a lhs.elems
#define b rhs.elems
    return FpExt(a[0] * b[0] + NBETA * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]),
                 a[0] * b[1] + a[1] * b[0] + NBETA * (a[2] * b[3] + a[3] * b[2]),
                 a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + NBETA * (a[3] * b[3]),
                 a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0]);
#undef a
#undef b
  }

  constexpr FpExt operator*=(FpExt rhs) {
    *this = *this * rhs;
    return *this;
  }

  constexpr FpExt operator*=(FpExt rhs) device {
    *this = FpExt(*this) * rhs;
    return *this;
  }

  constexpr FpExt operator*=(FpExt rhs) threadgroup {
    *this = FpExt(*this) * rhs;
    return *this;
  }

  // Equality
  friend constexpr bool operator==(FpExt lhs, FpExt rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      if (lhs.elems[i] != rhs.elems[i]) {
        return false;
      }
    }
    return true;
  }

  friend constexpr bool operator!=(FpExt lhs, FpExt rhs) { return !(lhs == rhs); }

  constexpr Fp constPart() const { return elems[0]; }

  constexpr Fp constPart() device const { return elems[0]; }
};

/// Raise an FpExt to a power
constexpr inline FpExt pow(FpExt x, size_t n) {
  FpExt tot(1);
  while (n != 0) {
    if (n % 2 == 1) {
      tot *= x;
    }
    n = n / 2;
    x *= x;
  }
  return tot;
}

/// Compute the multiplicative inverse of an FpExt.
constexpr inline FpExt inv(FpExt in) {
#define a in.elems
  // Compute the multiplicative inverse by basically looking at FpExt as a composite field and using
  // the same basic methods used to invert complex numbers.  We imagine that initially we have a
  // numerator of 1, and a denominator of a. i.e out = 1 / a; We set a' to be a with the first and
  // third components negated.  We then multiply the numerator and the denominator by a', producing
  // out = a' / (a * a'). By construction (a * a') has 0's in it's first and third elements.  We
  // call this number, 'b' and compute it as follows.
  Fp b0 = a[0] * a[0] + BETA * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
  Fp b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + BETA * (a[3] * a[3]);
  // Now, we make b' by inverting b2.  When we multiply both sizes by b', we get out = (a' * b') /
  // (b * b').  But by construction b * b' is in fact an element of Fp, call it c.
  Fp c = b0 * b0 + BETA * b2 * b2;
  // But we can now invert C directly, and multiply by a'*b', out = a'*b'*inv(c)
  Fp ic = inv(c);
  // Note: if c == 0 (really should only happen if in == 0), our 'safe' version of inverse results
  // in ic == 0, and thus out = 0, so we have the same 'safe' behavior for FpExt.  Oh, and since we
  // want to multiply everything by ic, it's slightly faster to premultiply the two parts of b by ic
  // (2 multiplies instead of 4)
  b0 *= ic;
  b2 *= ic;
  return FpExt(a[0] * b0 + BETA * a[2] * b2,
               -a[1] * b0 + NBETA * a[3] * b2,
               -a[0] * b2 + a[2] * b0,
               a[1] * b2 - a[3] * b0);
#undef a
}

#undef BETA
#undef NBETA
