#[inline]
pub fn pair_in_order<T: Ord>((x, y): (T, T)) -> (T, T) {
    if x < y {
        (x, y)
    } else {
        (y, x)
    }
}
