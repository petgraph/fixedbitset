#[macro_use]
extern crate criterion;
extern crate fixedbitset;

use criterion::Criterion;
use fixedbitset::{FixedBitSet};
use std::mem::size_of;

#[inline]
fn iter_ones_using_contains<F: FnMut(usize)>(fb: &FixedBitSet, f: &mut F) {
    for bit in 0 .. fb.len() {
       if fb.contains(bit) {
           f(bit);
       }
    }
}

#[inline]
fn iter_ones_using_slice_directly<F: FnMut(usize)>(fb: &FixedBitSet, f: &mut F) {
    for (block_idx, &block) in fb.as_slice().iter().enumerate() {
        let mut bit_pos = block_idx * size_of::<u32>() * 8;
        let mut block: u32 = block;

        while block != 0 {
            if (block & 1) == 1 {
                f(bit_pos);
            }
            block = block >> 1;
            bit_pos += 1;
        }
    }
}

fn bench_iter_ones_using_contains_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);

    c.bench_function("ones using contains: all zeros", move |b| b.iter(|| {
        let mut count = 0;
        iter_ones_using_contains(&fb, &mut |_bit| count += 1);
        count
    }));
}

fn bench_iter_ones_using_contains_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);
    fb.insert_range(..);

    c.bench_function("ones using contains: all ones", move |b| b.iter(|| {
        let mut count = 0;
        iter_ones_using_contains(&fb, &mut |_bit| count += 1);
        count
    }));
}

fn bench_iter_ones_using_slice_directly_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);

    c.bench_function("ones using slice: all zeros", move |b| b.iter(|| {
       let mut count = 0;
       iter_ones_using_slice_directly(&fb, &mut |_bit| count += 1);
       count
    }));
}

fn bench_iter_ones_using_slice_directly_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);
    fb.insert_range(..);

    c.bench_function("ones using slice: all ones", move |b| b.iter(|| {
       let mut count = 0;
       iter_ones_using_slice_directly(&fb, &mut |_bit| count += 1);
       count
    }));
}

fn bench_iter_ones_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);

    c.bench_function("ones: all zeros", move |b| b.iter(|| {
        let mut count = 0;
        for _ in fb.ones() {
            count += 1;
        }
        count
    }));
}

fn bench_iter_ones_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);
    fb.insert_range(..);

    c.bench_function("ones: all ones", move |b| b.iter(|| {
        let mut count = 0;
        for _ in fb.ones() {
            count += 1;
        }
        count
    }));
}

fn bench_insert_range(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);

    c.bench_function("insert range", move |b| b.iter(|| {
        fb.insert_range(..)
    }));
}

fn bench_insert_range_using_loop(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);

    c.bench_function("insert range using loop", move |b| b.iter(|| {
        for i in 0..N {
            fb.insert(i);
        }
    }));
}

criterion_group!(benches,
    bench_iter_ones_using_contains_all_zeros,
    bench_iter_ones_using_contains_all_ones,
    bench_iter_ones_using_slice_directly_all_zeros,
    bench_iter_ones_using_slice_directly_all_ones,
    bench_iter_ones_all_zeros,
    bench_iter_ones_all_ones,
    bench_insert_range,
    bench_insert_range_using_loop
);
criterion_main!(benches);
