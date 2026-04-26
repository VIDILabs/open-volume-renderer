// ======================================================================== //
// Unit tests for gdt math primitives as re-exported through ovr::math.     //
// These exercise the narrow slice of gdt that OVR actually depends on;    //
// a full gdt test suite lives upstream.                                    //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <math_def.h>

using namespace ovr;

TEST_CASE("vec2f: construction, arithmetic, comparison") {
  vec2f a{1.f, 2.f};
  vec2f b{3.f, 4.f};

  auto sum = a + b;
  CHECK(sum.x == doctest::Approx(4.f));
  CHECK(sum.y == doctest::Approx(6.f));

  auto diff = b - a;
  CHECK(diff.x == doctest::Approx(2.f));
  CHECK(diff.y == doctest::Approx(2.f));

  auto scaled = a * 2.f;
  CHECK(scaled.x == doctest::Approx(2.f));
  CHECK(scaled.y == doctest::Approx(4.f));
}

TEST_CASE("vec3f: length and normalize") {
  vec3f v{3.f, 0.f, 4.f};
  CHECK(length(v) == doctest::Approx(5.f));

  auto n = normalize(v);
  CHECK(length(n) == doctest::Approx(1.f).epsilon(1e-6));
  CHECK(n.x == doctest::Approx(0.6f));
  CHECK(n.z == doctest::Approx(0.8f));
}

TEST_CASE("vec3f: cross product is anti-commutative and orthogonal") {
  vec3f x{1.f, 0.f, 0.f};
  vec3f y{0.f, 1.f, 0.f};

  auto z = cross(x, y);
  CHECK(z.x == doctest::Approx(0.f));
  CHECK(z.y == doctest::Approx(0.f));
  CHECK(z.z == doctest::Approx(1.f));

  auto neg_z = cross(y, x);
  CHECK(neg_z.z == doctest::Approx(-1.f));
}

TEST_CASE("box3f: extend starts from inverted-infinity and grows correctly") {
  box3f b; // default: lower = +inf, upper = -inf (empty)
  CHECK(b.lower.x > b.upper.x); // invariant of an "empty" bounds

  b.extend(vec3f{1.f, 2.f, 3.f});
  CHECK(b.lower.x == doctest::Approx(1.f));
  CHECK(b.upper.x == doctest::Approx(1.f));

  b.extend(vec3f{-1.f, 10.f, 0.f});
  CHECK(b.lower.x == doctest::Approx(-1.f));
  CHECK(b.upper.y == doctest::Approx(10.f));
  CHECK(b.lower.z == doctest::Approx(0.f));
  CHECK(b.upper.z == doctest::Approx(3.f));
}

TEST_CASE("box3f: extend with another box produces the union") {
  box3f a;
  a.extend(vec3f{0.f, 0.f, 0.f});
  a.extend(vec3f{1.f, 1.f, 1.f});

  box3f b;
  b.extend(vec3f{2.f, -1.f, 0.5f});
  b.extend(vec3f{3.f, 0.5f, 2.f});

  a.extend(b);
  CHECK(a.lower.x == doctest::Approx(0.f));
  CHECK(a.upper.x == doctest::Approx(3.f));
  CHECK(a.lower.y == doctest::Approx(-1.f));
  CHECK(a.upper.y == doctest::Approx(1.f));
  CHECK(a.lower.z == doctest::Approx(0.f));
  CHECK(a.upper.z == doctest::Approx(2.f));
}

TEST_CASE("affine3f: identity composed with inverse is identity") {
  using math::affine3f;
  using math::linear3f;

  affine3f t = affine3f::translate(vec3f{1.f, 2.f, 3.f});
  affine3f inv = rcp(t);
  affine3f composed = t * inv;

  vec3f p{5.f, 6.f, 7.f};
  vec3f p2 = xfmPoint(composed, p);
  CHECK(p2.x == doctest::Approx(p.x).epsilon(1e-5));
  CHECK(p2.y == doctest::Approx(p.y).epsilon(1e-5));
  CHECK(p2.z == doctest::Approx(p.z).epsilon(1e-5));
}

TEST_CASE("clamp and min/max behave as expected") {
  CHECK(clamp(5.f, 0.f, 3.f) == doctest::Approx(3.f));
  CHECK(clamp(-1.f, 0.f, 3.f) == doctest::Approx(0.f));
  CHECK(clamp(1.5f, 0.f, 3.f) == doctest::Approx(1.5f));

  CHECK(min(1, 2) == 1);
  CHECK(max(1, 2) == 2);
}
