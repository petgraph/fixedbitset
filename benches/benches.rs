extern crate byteorder;
#[macro_use]
extern crate criterion;
extern crate rand;
extern crate fixedbitset;

use std::mem::size_of;
use criterion::{Criterion, Fun};
use rand::Rng;
use fixedbitset::{FixedBitSet};

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

fn make_bench_iter_ones(fb: &FixedBitSet) -> Vec<Fun<()>> {
    let default = {
        let fb = fb.clone();
        Fun::new("default", move |b, _| b.iter(|| {
            let mut count = 0;
            for _ in fb.ones() {
                count += 1;
            }
            count
        }))
    };
    let contains = {
        let fb = fb.clone();
        Fun::new("contains", move |b, _| b.iter(|| {
            let mut count = 0;
            iter_ones_using_contains(&fb, &mut |_bit| count += 1);
            count
        }))
    };
    let slice = {
       let fb = fb.clone();
       Fun::new("slice directly", move |b, _| b.iter(|| {
           let mut count = 0;
           iter_ones_using_slice_directly(&fb, &mut |_bit| count += 1);
           count
       }))
    };
    vec![default, contains, slice]
}

fn bench_iter_ones_all_zeros(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);
    c.bench_functions("iter ones: all zeros", make_bench_iter_ones(&fb), ());
}

fn bench_iter_ones_all_ones(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut fb = FixedBitSet::with_capacity(N);
    fb.insert_range(..);
    c.bench_functions("iter ones: all ones", make_bench_iter_ones(&fb), ());
}

fn bench_iter_ones_random(c: &mut Criterion) {
    const N: usize = 15625 * 2 * 32;
    let mut fb = FixedBitSet::with_capacity(N);
    let mut rng = rand::thread_rng();
    {
        let p = fb.as_mut_slice();
        for w in p {
            *w = rng.next_u32();
        }
    }
    assert!(fb.count_ones(..) > 10);
    c.bench_functions("iter ones: random", make_bench_iter_ones(&fb), ());
}

fn bench_insert_range(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let fb = FixedBitSet::with_capacity(N);

    let default = {
        let mut fb = fb.clone();
        Fun::new("default", move |b, _| b.iter(|| {
            fb.insert_range(..)
        }))
    };
    let loop_ = {
        let mut fb = fb.clone();
        Fun::new("loop", move |b, _| b.iter(|| {
            for i in 0..N {
                fb.insert(i);
            }
        }))
    };
    c.bench_functions("insert range", vec![default, loop_], ());
}

criterion_group!(benches,
    bench_iter_ones_all_zeros,
    bench_iter_ones_all_ones,
    bench_iter_ones_random,
    bench_insert_range
);
criterion_main!(benches);
