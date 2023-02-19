//! [The Pattern API] implementation for searching in `&[T]`.
//!
//! The implementation provides generic mechanism for using different pattern
//! types when searching through a slice.  Although this API is unstable, it is
//! exposed via stable APIs on the [`&[T]`] type.
//!
//! Depending on the type of the pattern, the behaviour of methods like
//! [`[T]::find`] and [`[T]::contains`] can change. The table below describes
//! some of those behaviours.
//!
//! | Pattern type             | Match condition       |
//! |--------------------------|-----------------------|
//! | `&T`                     | is contained in slice |
//! | `&[T]`                   | is subslice           |
//! | `&[T; N]`                | is subslice           |
//! | `&Vec<T>`                | is subslice           |
//!
//! Beware that slice patterns over a `&[T]` [haystack] perform an ‘is subslice’
//! match rather than ‘does any of characters equal’ match.  This may be
//! confusing since syntax for both is somewhat similar.  For example:
//!
//! ```
//! # #![feature(pattern)]
//!
//! let haystack: &str = "Quick brown fox";
//! assert_eq!(haystack.find(&['u', 'x']), Some(1));
//!     // haystack matches `‘u’ or ‘x’` on first position.
//!
//! let haystack: &[u8] = b"Quick brown fox";
//! assert_eq!(haystack.find(&[b'u', b'x']), None);
//!     // haystack doesn’t contain `‘ux’` subslice.
//! ```
//!
//! [The Pattern API]: crate::pattern
//! [haystack][crate::pattern::Haystack]

#![unstable(
    feature = "slice_pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "56345"
)]

use crate::mem::{replace, take};
use crate::pattern;
use crate::pattern::{Haystack, MatchOnly, Pattern, RejectOnly};

use super::cmp::SliceContains;

/////////////////////////////////////////////////////////////////////////////
// Impl for Haystack
/////////////////////////////////////////////////////////////////////////////

impl<'a, T> Haystack for &'a [T] {
    type Cursor = usize;

    fn cursor_at_front(self) -> usize {
        0
    }
    fn cursor_at_back(self) -> usize {
        self.len()
    }

    fn is_empty(self) -> bool {
        self.is_empty()
    }

    unsafe fn get_unchecked(self, range: core::ops::Range<usize>) -> Self {
        if cfg!(debug_assertions) {
            self.get(range).unwrap()
        } else {
            // SAFETY: Caller promises cursor is valid.
            unsafe { self.get_unchecked(range) }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl Pattern for &T
/////////////////////////////////////////////////////////////////////////////

/// Pattern implementation for searching for an element in a slice.
///
/// The pattern matches a single element in a slice.
///
/// # Examples
///
/// ```
/// # #![feature(pattern)]
///
/// let nums = &[10, 40, 30, 40];
/// assert_eq!(nums.find(&40), Some(1));
/// assert_eq!(nums.find(&42), None);
/// ```
impl<'hs, 'p, T: PartialEq> Pattern<&'hs [T]> for &'p T {
    type Searcher = ElementSearcher<'hs, 'p, T>;

    fn into_searcher(self, haystack: &'hs [T]) -> Self::Searcher {
        // FIXME: We probably should specialise this for u8 and i8 the same way
        // we specialise SliceContains
        Self::Searcher::new(haystack, self)
    }

    fn is_contained_in(self, haystack: &'hs [T]) -> bool {
        T::slice_contains_element(haystack, self)
    }

    fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
        haystack.first() == Some(self)
    }
    fn strip_prefix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        match haystack.split_first() {
            Some((first, tail)) if first == self => Some(tail),
            _ => None,
        }
    }

    fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
        haystack.last() == Some(self)
    }
    fn strip_suffix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        match haystack.split_last() {
            Some((last, head)) if last == self => Some(head),
            _ => None,
        }
    }
}

/// Pattern implementation for searching for an element in a slice.
///
/// The pattern matches a single element in a slice.
///
/// # Examples
///
/// ```
/// # #![feature(pattern)]
///
/// let nums = &[10, 40, 30, 40];
/// assert_eq!(nums.find(&40), Some(1));
/// assert_eq!(nums.find(&42), None);
/// ```
impl<'hs, 'o, 'p, T: PartialEq> Pattern<&'hs [T]> for &'o &'p T {
    type Searcher = ElementSearcher<'hs, 'p, T>;

    fn into_searcher(self, haystack: &'hs [T]) -> Self::Searcher {
        (*self).into_searcher(haystack)
    }

    fn is_contained_in(self, haystack: &'hs [T]) -> bool {
        (*self).is_contained_in(haystack)
    }

    fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
        (*self).is_prefix_of(haystack)
    }

    fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
        (*self).is_suffix_of(haystack)
    }

    fn strip_prefix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        (*self).strip_prefix_of(haystack)
    }

    fn strip_suffix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        (*self).strip_suffix_of(haystack)
    }
}

/// Associated type for `<&[T] as Pattern>::Searcher`.
#[derive(Clone, Debug)]
pub struct ElementSearcher<'hs, 'p, T> {
    haystack: &'hs [T],
    state: NeedleSearcherState<&'p T>,
}

impl<'hs, 'p, T: PartialEq> ElementSearcher<'hs, 'p, T> {
    fn new(haystack: &'hs [T], needle: &'p T) -> Self {
        let state = NeedleSearcherState::new(haystack.len(), needle);
        Self { haystack, state }
    }
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::Searcher<&'hs [T]> for ElementSearcher<'hs, 'p, T> {
    fn haystack(&self) -> &'hs [T] {
        self.haystack
    }

    fn next(&mut self) -> pattern::SearchStep {
        self.state.next_fwd(self.haystack)
    }
    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.state.next_fwd::<MatchOnly, _>(self.haystack).0
    }
    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.state.next_fwd::<RejectOnly, _>(self.haystack).0
    }
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::ReverseSearcher<&'hs [T]>
    for ElementSearcher<'hs, 'p, T>
{
    fn next_back(&mut self) -> pattern::SearchStep {
        self.state.next_bwd(self.haystack)
    }
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.state.next_bwd::<MatchOnly, _>(self.haystack).0
    }
    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.state.next_bwd::<RejectOnly, _>(self.haystack).0
    }
}

impl<'hs, 'p, T: PartialEq> pattern::DoubleEndedSearcher<&'hs [T]> for ElementSearcher<'hs, 'p, T> {}

/////////////////////////////////////////////////////////////////////////////
// Impl Pattern for &[T] and &[T; N]
/////////////////////////////////////////////////////////////////////////////

/// Pattern implementation for searching a subslice in a slice.
///
/// The pattern matches a subslice of a larger slice.  An empty pattern matches
/// around every element in a slice (including at the beginning and end of the
/// slice).
///
/// # Examples
///
/// ```
/// # #![feature(pattern)]
/// use core::pattern::{Pattern, Searcher};
///
/// // Simple usage
/// let nums: &[i32] = &[10, 40, 30, 40];
/// assert_eq!(nums.find(&[40]), Some(1));
/// assert_eq!(nums.find(&[40, 30]), Some(1));
/// assert_eq!(nums.find(&[42, 30]), None);
///
/// // Empty pattern
/// let empty: &[i32] = &[];
/// let mut s = empty.into_searcher(nums);
/// assert_eq!(s.next_match(), Some((0, 0)));
/// assert_eq!(s.next_match(), Some((1, 1)));
/// assert_eq!(s.next_match(), Some((2, 2)));
/// assert_eq!(s.next_match(), Some((3, 3)));
/// assert_eq!(s.next_match(), Some((4, 4)));
/// assert_eq!(s.next_match(), None);
///
/// // Difference with str patterns.
/// assert_eq!("Foo".find(&['f', 'o']), Some(1));
///             // -- "Foo" contains letter 'o' at index 1.
/// assert_eq!(b"Foo".find(&[b'f', b'o']), None);
///             // -- b"Foo" doesn’t contain subslice b"fo".
/// ```
impl<'hs, 'p, T: PartialEq> Pattern<&'hs [T]> for &'p [T] {
    type Searcher = SliceSearcher<'hs, 'p, T>;

    fn into_searcher(self, haystack: &'hs [T]) -> Self::Searcher {
        SliceSearcher::new(haystack, self)
    }

    fn is_contained_in(self, haystack: &'hs [T]) -> bool {
        if self.len() == 0 {
            true
        } else if self.len() == 1 {
            T::slice_contains_element(haystack, &self[0])
        } else if self.len() < haystack.len() {
            T::slice_contains_slice(haystack, self)
        } else if self.len() == haystack.len() {
            self == haystack
        } else {
            false
        }
    }

    fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
        haystack.get(..self.len()).map_or(false, |prefix| prefix == self)
    }
    fn strip_prefix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        self.is_prefix_of(haystack).then(|| {
            // SAFETY: prefix was just verified to exist.
            unsafe { haystack.get_unchecked(self.len()..) }
        })
    }

    fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
        haystack.len().checked_sub(self.len()).map_or(false, |n| &haystack[n..] == self)
    }
    fn strip_suffix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        self.is_suffix_of(haystack).then(|| {
            let n = haystack.len() - self.len();
            // SAFETY: suffix was just verified to exist.
            unsafe { haystack.get_unchecked(..n) }
        })
    }
}

/// Implements `Pattern<&'hs [T]>` for given type `$pattern` which must
/// implement `Index<RangeFull>` which converts the type into `&[T]` pattern.
#[macro_export]
#[unstable(feature = "slice_pattern_internals", issue = "none")]
macro_rules! impl_subslice_pattern {
    (
        $(#[$meta:meta])*
        ($($bounds:tt)*) for $pattern:ty
    ) => {
        $(#[$meta])*
        impl<'hs, $($bounds)*, T: $crate::cmp::PartialEq> $crate::pattern::Pattern<&'hs [T]> for $pattern {
            type Searcher = $crate::slice::pattern::SliceSearcher<'hs, 'p, T>;

            fn into_searcher(self, haystack: &'hs [T]) -> Self::Searcher {
                <&[T]>::into_searcher(&self[..], haystack)
            }

            fn is_contained_in(self, haystack: &'hs [T]) -> bool {
                <&[T]>::is_contained_in(&self[..], haystack)
            }

            fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
                <&[T]>::is_prefix_of(&self[..], haystack)
            }
            fn strip_prefix_of(self, haystack: &'hs [T]) -> $crate::option::Option<&'hs [T]> {
                <&[T]>::strip_prefix_of(&self[..], haystack)
            }

            fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
                <&[T]>::is_suffix_of(&self[..], haystack)
            }
            fn strip_suffix_of(self, haystack: &'hs [T]) -> $crate::option::Option<&'hs [T]> {
                <&[T]>::strip_suffix_of(&self[..], haystack)
            }
        }
    }
}

pub use impl_subslice_pattern;

impl_subslice_pattern! {
    /// Pattern implementation for searching a subslice in a slice.
    ///
    /// This is identical to a slice pattern: the pattern matches a subslice of
    /// a larger slice.  An empty array matches around every character in a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(pattern)]
    ///
    /// let slice: &[u8] = b"The quick brown fox";
    /// assert_eq!(slice.find(b"quick"), Some(4));
    /// assert_eq!(slice.find(b"slow"), None);
    /// assert_eq!(slice.find(b""), Some(0));
    /// ```
    ('p, const N: usize) for &'p [T; N]
}

impl_subslice_pattern! {
    /// Pattern implementation for searching a subslice in a slice.
    ///
    /// This is identical to a slice pattern: the pattern matches a subslice of
    /// a larger slice.  An empty array matches around every character in a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(pattern)]
    ///
    /// let slice: &[u8] = b"The quick brown fox";
    ///
    /// let pattern: &[u8] = b"quick";
    /// assert_eq!(slice.find(&pattern), Some(4));
    ///
    /// let pattern: &[u8] = b"slow";
    /// assert_eq!(slice.find(&pattern), None);
    ///
    /// let pattern: &[u8] = b"";
    /// assert_eq!(slice.find(&pattern), Some(0));
    /// ```
    ('o, 'p) for &'o &'p [T]
}

impl_subslice_pattern! {
    /// Pattern implementation for searching a subslice in a slice.
    ///
    /// This is identical to a slice pattern: the pattern matches a subslice of
    /// a larger slice.  An empty array matches around every character in a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(pattern)]
    ///
    /// let slice: &[u8] = b"The quick brown fox";
    /// assert_eq!(slice.find(&b"quick"), Some(4));
    /// assert_eq!(slice.find(&b"slow"), None);
    /// assert_eq!(slice.find(&b""), Some(0));
    /// ```
    ('o, 'p, const N: usize) for &'o &'p [T; N]
}

/// Associated type for `<&'p [T] as Pattern<&'hs [T]>>::Searcher`.
#[derive(Clone, Debug)]
pub struct SliceSearcher<'hs, 'p, T> {
    haystack: &'hs [T],
    state: SearcherState<'p, T>,
}

#[derive(Clone, Debug)]
enum SearcherState<'p, T> {
    Empty(EmptySearcherState),
    Element(NeedleSearcherState<&'p T>),
    Slice(NeedleSearcherState<&'p [T]>),
}

impl<'hs, 'p, T: PartialEq> SliceSearcher<'hs, 'p, T> {
    fn new(haystack: &'hs [T], needle: &'p [T]) -> Self {
        let state = match needle.len() {
            0 => SearcherState::Empty(EmptySearcherState::new(haystack)),
            1 => SearcherState::Element(NeedleSearcherState::new(haystack.len(), &needle[0])),
            _ => SearcherState::Slice(NeedleSearcherState::new(haystack.len(), needle)),
        };
        Self { haystack, state }
    }
}

macro_rules! delegate {
    ($method:ident -> $ret:ty as $delegate:ident::<$r:ty>) => {
        fn $method(&mut self) -> $ret {
            match &mut self.state {
                SearcherState::Empty(state) => state.$delegate::<$r>().into(),
                SearcherState::Element(state) => state.$delegate::<$r, _>(self.haystack).into(),
                SearcherState::Slice(state) => state.$delegate::<$r, _>(self.haystack).into(),
            }
        }
    };
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::Searcher<&'hs [T]> for SliceSearcher<'hs, 'p, T> {
    fn haystack(&self) -> &'hs [T] {
        self.haystack
    }

    delegate!(next        -> pattern::SearchStep    as next_fwd::<pattern::SearchStep>);
    delegate!(next_match  -> Option<(usize, usize)> as next_fwd::<pattern::MatchOnly>);
    delegate!(next_reject -> Option<(usize, usize)> as next_fwd::<pattern::RejectOnly>);
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::ReverseSearcher<&'hs [T]>
    for SliceSearcher<'hs, 'p, T>
{
    delegate!(next_back        -> pattern::SearchStep    as next_bwd::<pattern::SearchStep>);
    delegate!(next_match_back  -> Option<(usize, usize)> as next_bwd::<pattern::MatchOnly>);
    delegate!(next_reject_back -> Option<(usize, usize)> as next_bwd::<pattern::RejectOnly>);
}

/////////////////////////////////////////////////////////////////////////////
// Searching for an empty pattern
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct EmptySearcherState(pattern::EmptyNeedleSearcher<usize>);

impl EmptySearcherState {
    fn new<T>(haystack: &[T]) -> Self {
        Self(pattern::EmptyNeedleSearcher::new(haystack))
    }

    fn next_fwd<R: pattern::SearchResult>(&mut self) -> R {
        self.0.next_fwd(|range| range.start + 1)
    }
    fn next_bwd<R: pattern::SearchResult>(&mut self) -> R {
        self.0.next_bwd(|range| range.start + 1)
    }
}

/////////////////////////////////////////////////////////////////////////////
// Searching for a non-empty slice
/////////////////////////////////////////////////////////////////////////////

/// State for searching for a fixed-size subslices in a slice.
///
/// The state accepts generic [`Needle`] `N` as a pattern to be searched.
#[derive(Clone, Debug)]
struct NeedleSearcherState<N> {
    needle: N,
    start: usize,
    end: usize,
    is_match_fwd: bool,
    is_match_bwd: bool,
}

/// A fixed-length needle which can be searched in a slice.
trait Needle<'hs, T> {
    /// Returns length of the needle.  Must be positive.
    fn len(&self) -> usize;

    /// Returns whether needle matches in full given slice.
    ///
    /// `slice.len()` is guaranteed to equal `self.len()`.
    fn test(&mut self, slice: &'hs [T]) -> bool;
}

impl<'hs, T: PartialEq> Needle<'hs, T> for &T {
    fn len(&self) -> usize {
        1
    }
    fn test(&mut self, slice: &'hs [T]) -> bool {
        **self == slice[0]
    }
}

impl<'hs, T: PartialEq> Needle<'hs, T> for &[T] {
    fn len(&self) -> usize {
        <[_]>::len(*self)
    }
    fn test(&mut self, slice: &'hs [T]) -> bool {
        *self == slice
    }
}

impl<N> NeedleSearcherState<N> {
    /// Creates a new object searching for a `needle` in a haystack of given
    /// length.
    pub fn new(haystack_length: usize, needle: N) -> Self {
        Self { needle, start: 0, end: haystack_length, is_match_fwd: false, is_match_bwd: false }
    }

    /// Performs a forwards search and returns its result.
    ///
    /// If `R` is `SearchStep` and method returns a `SearchStep::Reject`, the
    /// reject will be longest possible reject, i.e. it’s going to end either at
    /// the start of following match or at the end of the yet-unexamined portion
    /// of the haystack.
    pub fn next_fwd<'hs, R, T>(&mut self, hs: &'hs [T]) -> R
    where
        N: Needle<'hs, T>,
        R: pattern::SearchResult,
    {
        if R::USE_EARLY_REJECT {
            return self.next_reject_fwd(hs).map_or(R::DONE, |(s, e)| R::rejecting(s, e).unwrap());
        }

        let needle_len = self.needle.len();
        let count = if self.start >= self.end {
            debug_assert_eq!(self.start, self.end);
            return R::DONE;
        } else if take(&mut self.is_match_fwd) {
            0
        } else if let Some(pos) =
            hs[self.start..self.end].windows(needle_len).position(|slice| self.needle.test(slice))
        {
            pos
        } else {
            let pos = replace(&mut self.start, self.end);
            return R::rejecting(pos, self.end).unwrap_or(R::DONE);
        };

        if count > 0 {
            let pos = self.start;
            self.start += count;
            if let Some(ret) = R::rejecting(pos, self.start) {
                self.is_match_fwd = true;
                return ret;
            } else if self.start >= self.end {
                return R::DONE;
            }
        }

        let pos = self.start;
        self.start += needle_len;
        R::matching(pos, self.start).unwrap()
    }

    /// Performs a forwards search and returns next shortest-possible reject.
    ///
    /// As an optimisation, the method won’t try to look for next match to
    /// figure out full extend of the reject portion of the haystack and instead
    /// will return shortest possible region.  This is based on the assumption
    /// that if user asks for the next reject they are really interested in
    /// where continuous series of matches ends.
    fn next_reject_fwd<'hs, T>(&mut self, hs: &'hs [T]) -> Option<(usize, usize)>
    where
        N: Needle<'hs, T>,
    {
        let needle_len = self.needle.len();
        if take(&mut self.is_match_fwd) && self.start < self.end {
            self.start += needle_len;
        }
        if let Some(n) =
            hs[self.start..self.end].chunks(needle_len).position(|slice| !self.needle.test(slice))
        {
            self.start += n * needle_len + 1;
            Some((self.start - 1, self.start))
        } else {
            let tail_len = (self.end - self.start) % needle_len;
            self.start = self.end;
            (tail_len > 0).then_some((self.end - tail_len, self.end))
        }
    }

    /// Performs a backwards search and returns its result.
    ///
    /// If `R` is `SearchStep` and method returns a `SearchStep::Reject`, the
    /// reject will be longest possible reject, i.e. it’s going to start either
    /// at the end of preceding match or at the of the start of the
    /// yet-unexamined portion of the haystack.
    pub fn next_bwd<'hs, R, T>(&mut self, hs: &'hs [T]) -> R
    where
        N: Needle<'hs, T>,
        R: pattern::SearchResult,
    {
        if R::USE_EARLY_REJECT {
            return self.next_reject_bwd(hs).map_or(R::DONE, |(s, e)| R::rejecting(s, e).unwrap());
        }

        let needle_len = self.needle.len();
        let count = if self.start >= self.end {
            debug_assert_eq!(self.start, self.end);
            return R::DONE;
        } else if take(&mut self.is_match_bwd) {
            0
        } else if let Some(pos) = hs[self.start..self.end]
            .windows(needle_len)
            .rev()
            .position(|slice| self.needle.test(slice))
        {
            pos
        } else {
            let pos = replace(&mut self.end, self.start);
            return R::rejecting(self.start, pos).unwrap_or(R::DONE);
        };

        if count > 0 {
            let pos = self.end;
            self.end -= count;
            if let Some(ret) = R::rejecting(self.end, pos) {
                self.is_match_bwd = true;
                return ret;
            } else if self.start >= self.end {
                return R::DONE;
            }
        }

        let pos = self.end;
        self.end -= needle_len;
        R::matching(self.end, pos).unwrap()
    }

    /// Performs a backwards search and returns next shortest-possible reject.
    ///
    /// As an optimisation, the method won’t try to look for next match to
    /// figure out full extend of the reject portion of the haystack and instead
    /// will return shortest possible region.  This is based on the assumption
    /// that if user asks for the next reject they are really interested in
    /// where continuous series of matches ends.
    fn next_reject_bwd<'hs, T>(&mut self, hs: &'hs [T]) -> Option<(usize, usize)>
    where
        N: Needle<'hs, T>,
    {
        let needle_len = self.needle.len();
        if take(&mut self.is_match_bwd) && self.start < self.end {
            self.end -= needle_len;
        }
        if let Some(n) =
            hs[self.start..self.end].rchunks(needle_len).position(|slice| !self.needle.test(slice))
        {
            self.end -= n * needle_len + 1;
            Some((self.end, self.end + 1))
        } else {
            let tail_len = (self.end - self.start) % needle_len;
            self.end = self.start;
            (tail_len > 0).then_some((self.start, self.start + tail_len))
        }
    }
}
