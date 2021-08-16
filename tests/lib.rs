extern crate fixedbitset;

use fixedbitset::FixedBitSet;
#[test]
fn it_works() {
    const N: usize = 50;
    let mut fb = FixedBitSet::with_capacity(N);

    for i in 0..(N + 10) {
        assert_eq!(fb.contains(i), false);
    }

    fb.insert(10);
    fb.set(11, false);
    fb.set(12, false);
    fb.set(12, true);
    fb.set(N - 1, true);
    assert!(fb.contains(10));
    assert!(!fb.contains(11));
    assert!(fb.contains(12));
    assert!(fb.contains(N - 1));
    for i in 0..N {
        let contain = i == 10 || i == 12 || i == N - 1;
        assert_eq!(contain, fb[i]);
    }

    fb.clear();
}

#[test]
fn with_blocks() {
    let fb = FixedBitSet::with_capacity_and_blocks(50, vec![8u32, 0u32]);
    assert!(fb.contains(3));

    let ones: Vec<_> = fb.ones().collect();
    assert_eq!(ones.len(), 1);
}

#[test]
fn with_blocks_too_small() {
    let mut fb = FixedBitSet::with_capacity_and_blocks(500, vec![8u32, 0u32]);
    fb.insert(400);
    assert!(fb.contains(400));
}

#[test]
fn with_blocks_too_big() {
    let fb = FixedBitSet::with_capacity_and_blocks(1, vec![8u32]);

    // since capacity is 1, 3 shouldn't be set here
    assert!(!fb.contains(3));
}

#[test]
fn with_blocks_too_big_range_check() {
    let fb = FixedBitSet::with_capacity_and_blocks(1, vec![0xff]);

    // since capacity is 1, only 0 should be set
    assert!(fb.contains(0));
    for i in 1..0xff {
        assert!(!fb.contains(i));
    }
}

#[test]
fn grow() {
    let mut fb = FixedBitSet::with_capacity(48);
    for i in 0..fb.len() {
        fb.set(i, true);
    }

    let old_len = fb.len();
    fb.grow(72);
    for j in 0..fb.len() {
        assert_eq!(fb.contains(j), j < old_len);
    }
    fb.set(64, true);
    assert!(fb.contains(64));
}

#[test]
fn test_toggle() {
    let mut fb = FixedBitSet::with_capacity(16);
    fb.toggle(1);
    fb.put(2);
    fb.toggle(2);
    fb.put(3);
    assert!(fb.contains(1));
    assert!(!fb.contains(2));
    assert!(fb.contains(3));
}

#[test]
fn copy_bit() {
    let mut fb = FixedBitSet::with_capacity(48);
    for i in 0..fb.len() {
        fb.set(i, true);
    }
    fb.set(42, false);
    fb.copy_bit(42, 2);
    assert!(!fb.contains(42));
    assert!(!fb.contains(2));
    assert!(fb.contains(1));
    fb.copy_bit(1, 42);
    assert!(fb.contains(42));
    fb.copy_bit(1024, 42);
    assert!(!fb[42]);
}

#[test]
fn count_ones() {
    let mut fb = FixedBitSet::with_capacity(100);
    fb.set(11, true);
    fb.set(12, true);
    fb.set(7, true);
    fb.set(35, true);
    fb.set(40, true);
    fb.set(77, true);
    fb.set(95, true);
    fb.set(50, true);
    fb.set(99, true);
    assert_eq!(fb.count_ones(..7), 0);
    assert_eq!(fb.count_ones(..8), 1);
    assert_eq!(fb.count_ones(..11), 1);
    assert_eq!(fb.count_ones(..12), 2);
    assert_eq!(fb.count_ones(..13), 3);
    assert_eq!(fb.count_ones(..35), 3);
    assert_eq!(fb.count_ones(..36), 4);
    assert_eq!(fb.count_ones(..40), 4);
    assert_eq!(fb.count_ones(..41), 5);
    assert_eq!(fb.count_ones(50..), 4);
    assert_eq!(fb.count_ones(70..95), 1);
    assert_eq!(fb.count_ones(70..96), 2);
    assert_eq!(fb.count_ones(70..99), 2);
    assert_eq!(fb.count_ones(..), 9);
    assert_eq!(fb.count_ones(0..100), 9);
    assert_eq!(fb.count_ones(0..0), 0);
    assert_eq!(fb.count_ones(100..100), 0);
    assert_eq!(fb.count_ones(7..), 9);
    assert_eq!(fb.count_ones(8..), 8);
}

#[test]
fn ones() {
    let mut fb = FixedBitSet::with_capacity(100);
    fb.set(11, true);
    fb.set(12, true);
    fb.set(7, true);
    fb.set(35, true);
    fb.set(40, true);
    fb.set(77, true);
    fb.set(95, true);
    fb.set(50, true);
    fb.set(99, true);

    let ones: Vec<_> = fb.ones().collect();

    assert_eq!(vec![7, 11, 12, 35, 40, 50, 77, 95, 99], ones);
}

#[test]
fn iter_ones_range() {
    fn test_range(from: usize, to: usize, capa: usize) {
        assert!(to <= capa);
        let mut fb = FixedBitSet::with_capacity(capa);
        for i in from..to {
            fb.insert(i);
        }
        let ones: Vec<_> = fb.ones().collect();
        let expected: Vec<_> = (from..to).collect();
        assert_eq!(expected, ones);
    }

    for i in 0..100 {
        test_range(i, 100, 100);
        test_range(0, i, 100);
    }
}

#[should_panic]
#[test]
fn count_ones_oob() {
    let fb = FixedBitSet::with_capacity(100);
    fb.count_ones(90..101);
}

#[should_panic]
#[test]
fn count_ones_negative_range() {
    let fb = FixedBitSet::with_capacity(100);
    fb.count_ones(90..80);
}

#[test]
fn count_ones_panic() {
    for i in 1..128 {
        let fb = FixedBitSet::with_capacity(i);
        for j in 0..fb.len() + 1 {
            for k in j..fb.len() + 1 {
                assert_eq!(fb.count_ones(j..k), 0);
            }
        }
    }
}

#[test]
fn default() {
    let fb = FixedBitSet::default();
    assert_eq!(fb.len(), 0);
}

#[test]
fn insert_range() {
    let mut fb = FixedBitSet::with_capacity(97);
    fb.insert_range(..3);
    fb.insert_range(9..32);
    fb.insert_range(37..81);
    fb.insert_range(90..);
    for i in 0..97 {
        assert_eq!(
            fb.contains(i),
            i < 3 || 9 <= i && i < 32 || 37 <= i && i < 81 || 90 <= i
        );
    }
    assert!(!fb.contains(97));
    assert!(!fb.contains(127));
    assert!(!fb.contains(128));
}

#[test]
fn set_range() {
    let mut fb = FixedBitSet::with_capacity(48);
    fb.insert_range(..);

    fb.set_range(..32, false);
    fb.set_range(37.., false);
    fb.set_range(5..9, true);
    fb.set_range(40..40, true);

    for i in 0..48 {
        assert_eq!(fb.contains(i), 5 <= i && i < 9 || 32 <= i && i < 37);
    }
    assert!(!fb.contains(48));
    assert!(!fb.contains(64));
}

#[test]
fn toggle_range() {
    let mut fb = FixedBitSet::with_capacity(40);
    fb.insert_range(..10);
    fb.insert_range(34..38);

    fb.toggle_range(5..12);
    fb.toggle_range(30..);

    for i in 0..40 {
        assert_eq!(
            fb.contains(i),
            i < 5 || 10 <= i && i < 12 || 30 <= i && i < 34 || 38 <= i
        );
    }
    assert!(!fb.contains(40));
    assert!(!fb.contains(64));
}

#[test]
fn bitand_equal_lengths() {
    let len = 109;
    let a_end = 59;
    let b_start = 23;
    let mut a = FixedBitSet::with_capacity(len);
    let mut b = FixedBitSet::with_capacity(len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a & &b;
    for i in 0..b_start {
        assert!(!ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(ab.contains(i));
    }
    for i in a_end..len {
        assert!(!ab.contains(i));
    }
    assert_eq!(a.len(), ab.len());
}

#[test]
fn bitand_first_smaller() {
    let a_len = 113;
    let b_len = 137;
    let len = std::cmp::min(a_len, b_len);
    let a_end = 97;
    let b_start = 89;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a & &b;
    for i in 0..b_start {
        assert!(!ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(ab.contains(i));
    }
    for i in a_end..len {
        assert!(!ab.contains(i));
    }
    assert_eq!(a.len(), ab.len());
}

#[test]
fn bitand_first_larger() {
    let a_len = 173;
    let b_len = 137;
    let len = std::cmp::min(a_len, b_len);
    let a_end = 107;
    let b_start = 43;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a & &b;
    for i in 0..b_start {
        assert!(!ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(ab.contains(i));
    }
    for i in a_end..len {
        assert!(!ab.contains(i));
    }
    assert_eq!(b.len(), ab.len());
}

#[test]
fn intersection() {
    let len = 109;
    let a_end = 59;
    let b_start = 23;
    let mut a = FixedBitSet::with_capacity(len);
    let mut b = FixedBitSet::with_capacity(len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);

    let mut ab = a.intersection(&b).collect::<FixedBitSet>();

    for i in 0..b_start {
        assert!(!ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(ab.contains(i));
    }
    for i in a_end..len {
        assert!(!ab.contains(i));
    }

    a.intersect_with(&b);
    // intersection + collect produces the same results but with a shorter length.
    ab.grow(a.len());
    assert_eq!(
        ab, a,
        "intersection and intersect_with produce the same results"
    );
}

#[test]
fn union() {
    let a_len = 173;
    let b_len = 137;
    let a_start = 139;
    let b_end = 107;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(a_start.., true);
    b.set_range(..b_end, true);
    let ab = a.union(&b).collect::<FixedBitSet>();
    for i in a_start..a_len {
        assert!(ab.contains(i));
    }
    for i in 0..b_end {
        assert!(ab.contains(i));
    }
    for i in b_end..a_start {
        assert!(!ab.contains(i));
    }

    a.union_with(&b);
    assert_eq!(ab, a, "union and union_with produce the same results");
}

#[test]
fn difference() {
    let a_len = 83;
    let b_len = 151;
    let a_start = 0;
    let a_end = 79;
    let b_start = 53;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(a_start..a_end, true);
    b.set_range(b_start..b_len, true);
    let mut a_diff_b = a.difference(&b).collect::<FixedBitSet>();
    for i in a_start..b_start {
        assert!(a_diff_b.contains(i));
    }
    for i in b_start..b_len {
        assert!(!a_diff_b.contains(i));
    }

    a.difference_with(&b);
    // difference + collect produces the same results but with a shorter length.
    a_diff_b.grow(a.len());
    assert_eq!(
        a_diff_b, a,
        "difference and difference_with produce the same results"
    );
}

#[test]
fn symmetric_difference() {
    let a_len = 83;
    let b_len = 151;
    let a_start = 47;
    let a_end = 79;
    let b_start = 53;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(a_start..a_end, true);
    b.set_range(b_start..b_len, true);
    let a_sym_diff_b = a.symmetric_difference(&b).collect::<FixedBitSet>();
    for i in 0..a_start {
        assert!(!a_sym_diff_b.contains(i));
    }
    for i in a_start..b_start {
        assert!(a_sym_diff_b.contains(i));
    }
    for i in b_start..a_end {
        assert!(!a_sym_diff_b.contains(i));
    }
    for i in a_end..b_len {
        assert!(a_sym_diff_b.contains(i));
    }

    a.symmetric_difference_with(&b);
    assert_eq!(
        a_sym_diff_b, a,
        "symmetric_difference and _with produce the same results"
    );
}

#[test]
fn bitor_equal_lengths() {
    let len = 109;
    let a_start = 17;
    let a_end = 23;
    let b_start = 19;
    let b_end = 59;
    let mut a = FixedBitSet::with_capacity(len);
    let mut b = FixedBitSet::with_capacity(len);
    a.set_range(a_start..a_end, true);
    b.set_range(b_start..b_end, true);
    let ab = &a | &b;
    for i in 0..a_start {
        assert!(!ab.contains(i));
    }
    for i in a_start..b_end {
        assert!(ab.contains(i));
    }
    for i in b_end..len {
        assert!(!ab.contains(i));
    }
    assert_eq!(ab.len(), len);
}

#[test]
fn bitor_first_smaller() {
    let a_len = 113;
    let b_len = 137;
    let a_end = 89;
    let b_start = 97;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a | &b;
    for i in 0..a_end {
        assert!(ab.contains(i));
    }
    for i in a_end..b_start {
        assert!(!ab.contains(i));
    }
    for i in b_start..b_len {
        assert!(ab.contains(i));
    }
    assert_eq!(b_len, ab.len());
}

#[test]
fn bitor_first_larger() {
    let a_len = 173;
    let b_len = 137;
    let a_start = 139;
    let b_end = 107;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(a_start.., true);
    b.set_range(..b_end, true);
    let ab = &a | &b;
    for i in a_start..a_len {
        assert!(ab.contains(i));
    }
    for i in 0..b_end {
        assert!(ab.contains(i));
    }
    for i in b_end..a_start {
        assert!(!ab.contains(i));
    }
    assert_eq!(a_len, ab.len());
}

#[test]
fn bitxor_equal_lengths() {
    let len = 109;
    let a_end = 59;
    let b_start = 23;
    let mut a = FixedBitSet::with_capacity(len);
    let mut b = FixedBitSet::with_capacity(len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a ^ &b;
    for i in 0..b_start {
        assert!(ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(!ab.contains(i));
    }
    for i in a_end..len {
        assert!(ab.contains(i));
    }
    assert_eq!(a.len(), ab.len());
}

#[test]
fn bitxor_first_smaller() {
    let a_len = 113;
    let b_len = 137;
    let len = std::cmp::max(a_len, b_len);
    let a_end = 97;
    let b_start = 89;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a ^ &b;
    for i in 0..b_start {
        assert!(ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(!ab.contains(i));
    }
    for i in a_end..len {
        assert!(ab.contains(i));
    }
    assert_eq!(b.len(), ab.len());
}

#[test]
fn bitxor_first_larger() {
    let a_len = 173;
    let b_len = 137;
    let len = std::cmp::max(a_len, b_len);
    let a_end = 107;
    let b_start = 43;
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.set_range(..a_end, true);
    b.set_range(b_start.., true);
    let ab = &a ^ &b;
    for i in 0..b_start {
        assert!(ab.contains(i));
    }
    for i in b_start..a_end {
        assert!(!ab.contains(i));
    }
    for i in a_end..b_len {
        assert!(ab.contains(i));
    }
    for i in b_len..len {
        assert!(!ab.contains(i));
    }
    assert_eq!(a.len(), ab.len());
}

#[test]
fn bitand_assign_shorter() {
    let a_ones: Vec<usize> = vec![2, 3, 7, 19, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let b_ones: Vec<usize> = vec![2, 7, 8, 11, 23, 31, 32];
    let a_and_b: Vec<usize> = vec![2, 7, 31, 32];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    a &= b;
    let res = a.ones().collect::<Vec<usize>>();
    assert!(res == a_and_b);
}

#[test]
fn bitand_assign_longer() {
    let a_ones: Vec<usize> = vec![2, 7, 8, 11, 23, 31, 32];
    let b_ones: Vec<usize> = vec![2, 3, 7, 19, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let a_and_b: Vec<usize> = vec![2, 7, 31, 32];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    a &= b;
    let res = a.ones().collect::<Vec<usize>>();
    assert!(res == a_and_b);
}

#[test]
fn bitor_assign_shorter() {
    let a_ones: Vec<usize> = vec![2, 3, 7, 19, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let b_ones: Vec<usize> = vec![2, 7, 8, 11, 23, 31, 32];
    let a_or_b: Vec<usize> = vec![2, 3, 7, 8, 11, 19, 23, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    a |= b;
    let res = a.ones().collect::<Vec<usize>>();
    assert!(res == a_or_b);
}

#[test]
fn bitor_assign_longer() {
    let a_ones: Vec<usize> = vec![2, 7, 8, 11, 23, 31, 32];
    let b_ones: Vec<usize> = vec![2, 3, 7, 19, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let a_or_b: Vec<usize> = vec![2, 3, 7, 8, 11, 19, 23, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    a |= b;
    let res = a.ones().collect::<Vec<usize>>();
    assert!(res == a_or_b);
}

#[test]
fn bitxor_assign_shorter() {
    let a_ones: Vec<usize> = vec![2, 3, 7, 19, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let b_ones: Vec<usize> = vec![2, 7, 8, 11, 23, 31, 32];
    let a_xor_b: Vec<usize> = vec![3, 8, 11, 19, 23, 37, 41, 43, 47, 71, 73, 101];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    a ^= b;
    let res = a.ones().collect::<Vec<usize>>();
    assert!(res == a_xor_b);
}

#[test]
fn bitxor_assign_longer() {
    let a_ones: Vec<usize> = vec![2, 7, 8, 11, 23, 31, 32];
    let b_ones: Vec<usize> = vec![2, 3, 7, 19, 31, 32, 37, 41, 43, 47, 71, 73, 101];
    let a_xor_b: Vec<usize> = vec![3, 8, 11, 19, 23, 37, 41, 43, 47, 71, 73, 101];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    a ^= b;
    let res = a.ones().collect::<Vec<usize>>();
    assert!(res == a_xor_b);
}

#[test]
fn op_assign_ref() {
    let mut a = FixedBitSet::with_capacity(8);
    let b = FixedBitSet::with_capacity(8);

    //check that all assign type operators work on references
    a &= &b;
    a |= &b;
    a ^= &b;
}

#[test]
fn subset_superset_shorter() {
    let a_ones: Vec<usize> = vec![7, 31, 32, 63];
    let b_ones: Vec<usize> = vec![2, 7, 19, 31, 32, 37, 41, 43, 47, 63, 73, 101];
    let mut a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let b = b_ones.iter().cloned().collect::<FixedBitSet>();
    assert!(a.is_subset(&b) && b.is_superset(&a));
    a.insert(14);
    assert!(!a.is_subset(&b) && !b.is_superset(&a));
}

#[test]
fn subset_superset_longer() {
    let a_len = 153;
    let b_len = 75;
    let a_ones: Vec<usize> = vec![7, 31, 32, 63];
    let b_ones: Vec<usize> = vec![2, 7, 19, 31, 32, 37, 41, 43, 47, 63, 73];
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.extend(a_ones.iter().cloned());
    b.extend(b_ones.iter().cloned());
    assert!(a.is_subset(&b) && b.is_superset(&a));
    a.insert(100);
    assert!(!a.is_subset(&b) && !b.is_superset(&a));
}

#[test]
fn is_disjoint_first_shorter() {
    let a_len = 75;
    let b_len = 153;
    let a_ones: Vec<usize> = vec![2, 19, 32, 37, 41, 43, 47, 73];
    let b_ones: Vec<usize> = vec![7, 23, 31, 63, 124];
    let mut a = FixedBitSet::with_capacity(a_len);
    let mut b = FixedBitSet::with_capacity(b_len);
    a.extend(a_ones.iter().cloned());
    b.extend(b_ones.iter().cloned());
    assert!(a.is_disjoint(&b));
    a.insert(63);
    assert!(!a.is_disjoint(&b));
}

#[test]
fn is_disjoint_first_longer() {
    let a_ones: Vec<usize> = vec![2, 19, 32, 37, 41, 43, 47, 73, 101];
    let b_ones: Vec<usize> = vec![7, 23, 31, 63];
    let a = a_ones.iter().cloned().collect::<FixedBitSet>();
    let mut b = b_ones.iter().cloned().collect::<FixedBitSet>();
    assert!(a.is_disjoint(&b));
    b.insert(2);
    assert!(!a.is_disjoint(&b));
}

#[test]
fn extend_on_empty() {
    let items: Vec<usize> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 29, 31, 37, 167];
    let mut fbs = FixedBitSet::with_capacity(0);
    fbs.extend(items.iter().cloned());
    let ones = fbs.ones().collect::<Vec<usize>>();
    assert!(ones == items);
}

#[test]
fn extend() {
    let items: Vec<usize> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 29, 31, 37, 167];
    let mut fbs = FixedBitSet::with_capacity(168);
    let new: Vec<usize> = vec![7, 37, 67, 137];
    for i in &new {
        fbs.put(*i);
    }

    fbs.extend(items.iter().cloned());

    let ones = fbs.ones().collect::<Vec<usize>>();
    let expected = {
        let mut tmp = items.clone();
        tmp.extend(new);
        tmp.sort();
        tmp.dedup();
        tmp
    };

    assert!(ones == expected);
}

#[test]
fn from_iterator() {
    let items: Vec<usize> = vec![0, 2, 4, 6, 8];
    let fb = items.iter().cloned().collect::<FixedBitSet>();
    for i in items {
        assert!(fb.contains(i));
    }
    for i in vec![1, 3, 5, 7] {
        assert!(!fb.contains(i));
    }
    assert_eq!(fb.len(), 9);
}

#[test]
fn from_iterator_ones() {
    let len = 257;
    let mut fb = FixedBitSet::with_capacity(len);
    for i in (0..len).filter(|i| i % 7 == 0) {
        fb.put(i);
    }
    fb.put(len - 1);
    let dup = fb.ones().collect::<FixedBitSet>();
    assert_eq!(fb.len(), dup.len());
    assert_eq!(
        fb.ones().collect::<Vec<usize>>(),
        dup.ones().collect::<Vec<usize>>()
    );
}

#[cfg(feature = "std")]
#[test]
fn binary_trait() {
    let items: Vec<usize> = vec![1, 5, 7, 10, 14, 15];
    let fb = items.iter().cloned().collect::<FixedBitSet>();

    assert_eq!(format!("{:b}", fb), "0100010100100011");
    assert_eq!(format!("{:#b}", fb), "0b0100010100100011");
}

#[cfg(feature = "std")]
#[test]
fn display_trait() {
    let len = 8;
    let mut fb = FixedBitSet::with_capacity(len);

    fb.put(4);
    fb.put(2);

    assert_eq!(format!("{}", fb), "00101000");
    assert_eq!(format!("{:#}", fb), "0b00101000");
}

#[test]
#[cfg(feature = "serde")]
fn test_serialize() {
    let mut fb = FixedBitSet::with_capacity(10);
    fb.put(2);
    fb.put(3);
    fb.put(6);
    fb.put(8);
    let serialized = serde_json::to_string(&fb).unwrap();
    assert_eq!(r#"{"data":[332],"length":10}"#, serialized);
}
