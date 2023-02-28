extern crate criterion;
extern crate fixedbitset;
use criterion::{criterion_group, criterion_main, Criterion};
use fixedbitset::FixedBitSet;
use std::hint::black_box;

#[inline]
fn iter_ones_using_contains<F: FnMut(usize)>(fb: &FixedBitSet, f: &mut F) {
    for bit in 0..fb.len() {
        if fb.contains(bit) {
            f(bit);
        }
    }
}

fn iter_ones_using_contains_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);

    c.bench_function("iter_ones/contains_all_zeros", |b| {
        b.iter(|| {
            let mut count = 0;
            iter_ones_using_contains(&fb, &mut |_bit| count += 1);
            count
        })
    });
}

fn iter_ones_using_contains_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);
    fb.insert_range(..);

    c.bench_function("iter_ones/contains_all_ones", |b| {
        b.iter(|| {
            let mut count = 0;
            iter_ones_using_contains(&fb, &mut |_bit| count += 1);
            count
        })
    });
}

fn iter_ones_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);

    c.bench_function("iter_ones/all_zeros", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in fb.ones() {
                count += 1;
            }
            count
        })
    });
}

fn iter_ones_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);
    fb.insert_range(..);

    c.bench_function("iter_ones/all_ones", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in fb.ones() {
                count += 1;
            }
            count
        })
    });
}

fn insert_range(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);

    c.bench_function("insert_range/1m", |b| b.iter(|| fb.insert_range(..)));
}

fn insert(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);

    c.bench_function("insert/1m", |b| {
        b.iter(|| {
            for i in 0..N {
                fb.insert(i);
            }
        })
    });
}

fn grow_and_insert(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);

    c.bench_function("grow_and_insert", |b| {
        b.iter(|| {
            for i in 0..N {
                fb.grow_and_insert(i);
            }
        })
    });
}

fn union_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb_a = FixedBitSet::with_capacity(N);
    let fb_b = FixedBitSet::with_capacity(N);

    c.bench_function("union_with/1m", |b| b.iter(|| fb_a.union_with(&fb_b)));
}

fn union_with_unchecked(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb_a = FixedBitSet::with_capacity(N);
    let fb_b = FixedBitSet::with_capacity(N);

    c.bench_function("union_with_unchecked/1m", |b| b.iter(|| fb_a.union_with_unchecked(&fb_b)));
}

fn intersect_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb_a = FixedBitSet::with_capacity(N);
    let fb_b = FixedBitSet::with_capacity(N);

    c.bench_function("intersect_with/1m", |b| {
        b.iter(|| fb_a.intersect_with(&fb_b))
    });
}

fn difference_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb_a = FixedBitSet::with_capacity(N);
    let fb_b = FixedBitSet::with_capacity(N);

    c.bench_function("difference_with/1m", |b| {
        b.iter(|| fb_a.difference_with(&fb_b))
    });
}

fn symmetric_difference_with(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb_a = FixedBitSet::with_capacity(N);
    let fb_b = FixedBitSet::with_capacity(N);

    c.bench_function("symmetric_difference_with/1m", |b| {
        b.iter(|| fb_a.symmetric_difference_with(&fb_b))
    });
}

fn clear(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb_a = FixedBitSet::with_capacity(N);

    c.bench_function("clear/1m", |b| b.iter(|| fb_a.clear()));
}

fn count_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb_a = FixedBitSet::with_capacity(N);

    c.bench_function("count_ones/1m", |b| {
        b.iter(|| black_box(fb_a.count_ones(..)))
    });
}

criterion_group!(
    benches,
    iter_ones_using_contains_all_zeros,
    iter_ones_using_contains_all_ones,
    iter_ones_all_zeros,
    iter_ones_all_ones,
    insert_range,
    insert,
    intersect_with,
    difference_with,
    union_with,
    union_with_unchecked,
    symmetric_difference_with,
    count_ones,
    clear,
    grow_and_insert,
);
criterion_main!(benches);
