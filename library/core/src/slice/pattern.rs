#![unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]

use crate::pattern::{Haystack, Pattern, Predicate, SearchStep};
use crate::pattern;

use super::cmp::SliceContains;

/////////////////////////////////////////////////////////////////////////////
// Impl for Haystack
/////////////////////////////////////////////////////////////////////////////

impl<'a, T> Haystack for &'a [T] {
    type Cursor = usize;

    fn cursor_at_front(&self) -> usize { 0 }
    fn cursor_at_back(&self) -> usize { self.len() }

    unsafe fn split_at_cursor_unchecked(self, pos: usize) -> (Self, Self) {
        // SAFETY: Caller promises cursor is valid.
        unsafe { (self.get_unchecked(..pos), self.get_unchecked(pos..)) }
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
        // TODO: We probably should specialise this for u8 and i8 the same way
        // we specialise SliceContains
        Self::Searcher::new(haystack, self)
    }

    fn is_contained_in(self, haystack: &'hs [T]) -> bool {
        T::slice_contains_element(haystack, self)
    }

    fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
        haystack.first() == Some(self)
    }

    fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
        haystack.last() == Some(self)
    }

    fn strip_prefix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        match haystack.split_first() {
            Some((first, tail)) if first == self => Some(tail),
            _ => None,
        }
    }

    fn strip_suffix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        match haystack.split_last() {
            Some((last, head)) if last == self => Some(head),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ElementSearcher<'hs, 'p, T> {
    /// Haystack we’re searching in.
    haystack: &'hs [T],
    /// Element we’re searching for.
    needle: &'p T,
    /// Internal state of the searcher.
    state: PredicateSearchState,
}

impl<'hs, 'p, T> ElementSearcher<'hs, 'p, T> {
    fn new(haystack: &'hs [T], needle: &'p T) -> Self {
        Self {
            haystack,
            needle,
            state: PredicateSearchState::new(haystack.len())
        }
    }
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::Searcher<&'hs [T]> for ElementSearcher<'hs, 'p, T> {
    fn haystack(&self) -> &'hs [T] { self.haystack }

    fn next(&mut self) -> SearchStep<usize> {
        self.state.next(self.haystack, &mut |element| element == self.needle)
    }

    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.state.next_match(self.haystack, &mut |element| element == self.needle)
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.state.next_reject(self.haystack, &mut |element| element == self.needle)
    }
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::ReverseSearcher<&'hs [T]> for ElementSearcher<'hs, 'p, T> {
    fn next_back(&mut self) -> SearchStep<usize> {
        self.state.next_back(self.haystack, &mut |element| element == self.needle)
    }

    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.state.next_match_back(self.haystack, &mut |element| element == self.needle)
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.state.next_reject_back(self.haystack, &mut |element| element == self.needle)
    }
}

impl<'hs, 'p, T: PartialEq> pattern::DoubleEndedSearcher<&'hs [T]> for ElementSearcher<'hs, 'p, T> {}

/////////////////////////////////////////////////////////////////////////////
// Impl Pattern for Predicate
/////////////////////////////////////////////////////////////////////////////

/// Pattern implementation for searching for an element matching given
/// predicate.
///
/// # Examples
///
/// ```
/// # #![feature(pattern)]
/// use core::pattern::predicate;
///
/// let nums = &[10, 40, 30, 40];
/// assert_eq!(nums.find(predicate(|n| n % 3 == 0)), Some(2));
/// assert_eq!(nums.find(predicate(|n| n % 2 == 1)), None);
/// ```
impl<'hs, T, F: FnMut(&'hs T) -> bool> Pattern<&'hs [T]> for Predicate<&'hs T, F> {
    type Searcher = PredicateSearcher<'hs, T, F>;

    fn into_searcher(self, haystack: &'hs [T]) -> Self::Searcher {
        Self::Searcher::new(haystack, self)
    }

    fn is_contained_in(mut self, haystack: &'hs [T]) -> bool {
        haystack.iter().any(|element| self.test(element))
    }

    fn is_prefix_of(mut self, haystack: &'hs [T]) -> bool {
        haystack.first().filter(|element| self.test(element)).is_some()
    }

    fn is_suffix_of(mut self, haystack: &'hs [T]) -> bool {
        haystack.last().filter(|element| self.test(element)).is_some()
    }

    fn strip_prefix_of(mut self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        match haystack.split_first() {
            Some((first, tail)) if self.test(first) => Some(tail),
            _ => None,
        }
    }

    fn strip_suffix_of(mut self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        match haystack.split_last() {
            Some((last, head)) if self.test(last) => Some(head),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PredicateSearcher<'hs, T, F> {
    /// Haystack we’re searching in.
    haystack: &'hs [T],
    /// Predicate used to match elements.
    pred: Predicate<&'hs T, F>,
    /// Internal state of the searcher.
    state: PredicateSearchState,
}

impl<'hs, T, F> PredicateSearcher<'hs, T, F> {
    fn new(haystack: &'hs [T], pred: Predicate<&'hs T, F>) -> Self {
        let state = PredicateSearchState::new(haystack.len());
        Self { haystack, pred, state }
    }
}

unsafe impl<'hs, T, F: FnMut(&'hs T) -> bool> pattern::Searcher<&'hs [T]> for PredicateSearcher<'hs, T, F> {
    fn haystack(&self) -> &'hs [T] { self.haystack }

    fn next(&mut self) -> SearchStep<usize> {
        self.state.next(self.haystack, self.pred.as_fn())
    }

    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.state.next_match(self.haystack, self.pred.as_fn())
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.state.next_reject(self.haystack, self.pred.as_fn())
    }
}

unsafe impl<'hs, T, F: FnMut(&'hs T) -> bool> pattern::ReverseSearcher<&'hs [T]> for PredicateSearcher<'hs, T, F> {
    fn next_back(&mut self) -> SearchStep<usize> {
        self.state.next_back(self.haystack, self.pred.as_fn())
    }

    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.state.next_match_back(self.haystack, self.pred.as_fn())
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.state.next_reject_back(self.haystack, self.pred.as_fn())
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl Pattern for &[T] and &[T; N]
/////////////////////////////////////////////////////////////////////////////

/// Pattern implementation for searching a subslice in a slice.
///
/// The pattern matches a subslice of a larger slice.  An empty pattern matches
/// around every character in a slice.
///
/// Note: Other than with slice patterns matching `str`, this pattern matches
/// a subslice rather than a single element of haystack being equal to element
/// of the pattern.
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
    type Searcher = Searcher<'hs, 'p, T>;

    fn into_searcher(self, haystack: &'hs [T]) -> Self::Searcher {
        Searcher::new(haystack, self)
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
    #[inline]
    fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
        haystack.get(..self.len()).map_or(false, |prefix| prefix == self)
    }


    #[inline]
    fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
        haystack
            .len()
            .checked_sub(self.len())
            .map_or(false, |n| &haystack[n..] == self)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        self.is_prefix_of(haystack).then(|| {
            // SAFETY: prefix was just verified to exist.
            unsafe { haystack.get_unchecked(self.len()..) }
        })
    }

    #[inline]
    fn strip_suffix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        self.is_suffix_of(haystack).then(|| {
            let n = haystack.len() - self.len();
            // SAFETY: suffix was just verified to exist.
            unsafe { haystack.get_unchecked(..n) }
        })
    }
}

/// Pattern implementation for searching a subslice in a slice.
///
/// This is identical to a slice pattern: the pattern matches a subslice of
/// a larger slice.  An empty array matches around every character in a slice.
///
/// Note: Other than with slice patterns matching `str`, this pattern matches
/// a subslice rather than a single element of haystack being equal to element
/// of the pattern.
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
impl<'hs, 'p, T: PartialEq, const N: usize> Pattern<&'hs [T]> for &'p [T; N] {
    type Searcher = Searcher<'hs, 'p, T>;

    fn into_searcher(self, haystack: &'hs [T]) -> Searcher<'hs, 'p, T> {
        Searcher::new(haystack, &self[..])
    }

    #[inline(always)]
    fn is_contained_in(self, haystack: &'hs [T]) -> bool {
        (&self[..]).is_contained_in(haystack)
    }

    #[inline(always)]
    fn is_prefix_of(self, haystack: &'hs [T]) -> bool {
        (&self[..]).is_prefix_of(haystack)
    }

    #[inline(always)]
    fn is_suffix_of(self, haystack: &'hs [T]) -> bool {
        (&self[..]).is_suffix_of(haystack)
    }

    #[inline(always)]
    fn strip_prefix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        (&self[..]).strip_prefix_of(haystack)
    }

    #[inline(always)]
    fn strip_suffix_of(self, haystack: &'hs [T]) -> Option<&'hs [T]> {
        (&self[..]).strip_suffix_of(haystack)
    }
}

#[derive(Clone, Debug)]
/// Associated type for `<&'p [T] as Pattern<&'hs [T]>>::Searcher`.
pub struct Searcher<'hs, 'p, T> {
    /// Haystack we’re searching in.
    haystack: &'hs [T],
    /// Subslice we’re searching for.
    needle: &'p [T],
    /// Internal state of the searcher.
    state: SearcherState,
}

#[derive(Clone, Debug)]
enum SearcherState {
    Empty(EmptySearcherState),
    Element(PredicateSearchState),
    Naive(NaiveSearcherState),
}

impl<'hs, 'p, T: PartialEq> Searcher<'hs, 'p, T> {
    fn new(haystack: &'hs [T], needle: &'p [T]) -> Searcher<'hs, 'p, T> {
        let state = match needle.len() {
            0 => SearcherState::Empty(EmptySearcherState::new(haystack.len())),
            1 => SearcherState::Element(PredicateSearchState::new(haystack.len())),
            _ => SearcherState::Naive(NaiveSearcherState::new(haystack.len())),
        };
        Searcher { haystack, needle, state }
    }
}

macro_rules! delegate {
    ($method:ident -> $ret:ty) => {
        fn $method(&mut self) -> $ret {
            match &mut self.state {
                SearcherState::Empty(state) => state.$method(),
                SearcherState::Element(state) => state.$method(self.haystack, &mut |element| {
                    // SAFETY: SearcherState::Element is created if and only if
                    // needle.len() == 1.
                    element == unsafe { self.needle.get_unchecked(0) }
                }),
                SearcherState::Naive(state) => state.$method(self.haystack, self.needle),
            }
        }
    }
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::Searcher<&'hs [T]> for Searcher<'hs, 'p, T> {
    fn haystack(&self) -> &'hs [T] {
        self.haystack
    }

    delegate!(next -> SearchStep);
    delegate!(next_match -> Option<(usize, usize)>);
    delegate!(next_reject -> Option<(usize, usize)>);
}

unsafe impl<'hs, 'p, T: PartialEq> pattern::ReverseSearcher<&'hs [T]> for Searcher<'hs, 'p, T> {
    delegate!(next_back -> SearchStep);
    delegate!(next_match_back -> Option<(usize, usize)>);
    delegate!(next_reject_back -> Option<(usize, usize)>);
}

/////////////////////////////////////////////////////////////////////////////
// Searching for an empty pattern
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct EmptySearcherState {
    start: usize,
    end: usize,
    is_match_fw: bool,
    is_match_bw: bool,
    // Needed in case of an empty haystack, see #85462
    is_finished: bool,
}

impl EmptySearcherState {
    fn new(haystack_length: usize) -> Self {
        Self {
            start: 0,
            end: haystack_length,
            is_match_fw: true,
            is_match_bw: true,
            is_finished: false,
        }
    }

    fn next(&mut self) -> SearchStep {
        if self.is_finished {
            return SearchStep::Done;
        }
        let is_match = self.is_match_fw;
        self.is_match_fw = !self.is_match_fw;
        let pos = self.start;
        if is_match {
            SearchStep::Match(pos, pos)
        } else if self.start < self.end {
            self.start += 1;
            SearchStep::Reject(pos, pos + 1)
        } else {
            self.is_finished = true;
            SearchStep::Done
        }
    }

    fn next_back(&mut self) -> SearchStep<usize> {
        if self.is_finished {
            return SearchStep::Done;
        }
        let is_match = self.is_match_bw;
        self.is_match_bw = !self.is_match_bw;
        let end = self.end;
        if is_match {
            SearchStep::Match(end, end)
        } else if self.end <= self.start {
            self.is_finished = true;
            SearchStep::Done
        } else {
            self.end -= 1;
            SearchStep::Reject(end - 1, end)
        }
    }

    fn next_match(&mut self) -> Option<(usize, usize)> {
        pattern::loop_next::<true, _>(|| self.next())
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        pattern::loop_next::<false, _>(|| self.next())
    }

    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        pattern::loop_next::<true, _>(|| self.next_back())
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        pattern::loop_next::<false, _>(|| self.next_back())
    }
}

/////////////////////////////////////////////////////////////////////////////
// Searching for a single element
/////////////////////////////////////////////////////////////////////////////

/// State of a searcher which tests one element at a time using a provided
/// predicate.
///
/// Matches are always one-element long.  Rejects can be arbitrarily long.
#[derive(Clone, Debug)]
struct PredicateSearchState {
    /// Position to start searching from.  Updated as we find new matches.
    start: usize,
    /// Position to end searching at.  Updated as we find new matches.
    end: usize,
    /// If true, we’re finished searching or haystack[start] is a match.
    is_match_fw: bool,
    /// If true, we’re finished searching or haystack[end-1] is a match.
    is_match_bw: bool
}

impl PredicateSearchState {
    fn new(haystack_length: usize) -> Self {
        Self {
            start: 0,
            end: haystack_length,
            is_match_fw: false,
            is_match_bw: false,
        }
    }

    fn next<'hs, T, F>(&mut self, hs: &'hs [T], pred: &mut F) -> SearchStep<usize>
    where F: FnMut(&'hs T) -> bool,
    {
        if self.start >= self.end {
            return SearchStep::Done;
        }
        let count = if self.is_match_fw {
            self.is_match_fw = false;
            0
        } else {
            self.count(false, hs, pred)
        };
        if count == 0 {
            self.start += 1;
            SearchStep::Match(self.start - 1, self.start)
        } else {
            self.is_match_fw = true;
            let pos = self.start;
            self.start += count;
            SearchStep::Reject(pos, self.start)
        }
    }

    fn next_match<'hs, T, F>(&mut self, hs: &'hs [T], pred: &mut F) -> Option<(usize, usize)>
    where F: FnMut(&'hs T) -> bool,
    {
        pattern::loop_next::<true, _>(|| self.next(hs, pred))
    }

    fn next_reject<'hs, T, F>(&mut self, hs: &'hs [T], pred: &mut F) -> Option<(usize, usize)>
    where F: FnMut(&'hs T) -> bool,
    {
        if self.start >= self.end {
            return None;
        }

        if self.is_match_fw {
            self.start += 1;
        }
        self.start += self.count(true, hs, pred);

        let count = self.count(false, hs, pred);
        if count == 0 {
            None
        } else {
            self.is_match_fw = true;
            let pos = self.start;
            self.start += count;
            Some((pos, self.start))
        }
    }

    fn next_back<'hs, T, F>(&mut self, hs: &'hs [T], pred: &mut F) -> SearchStep<usize>
    where F: FnMut(&'hs T) -> bool,
    {
        if self.start >= self.end {
            return SearchStep::Done
        }
        let count = if self.is_match_bw {
            self.is_match_bw = false;
            0
        } else {
            self.count_back(false, hs, pred)
        };
        let pos = self.end;
        if count == 0 {
            self.end -= 1;
            SearchStep::Match(self.end, pos)
        } else {
            self.is_match_bw = true;
            self.end -= count;
            SearchStep::Reject(self.end, pos)
        }
    }

    fn next_match_back<'hs, T, F>(&mut self, hs: &'hs [T], pred: &mut F) -> Option<(usize, usize)>
    where F: FnMut(&'hs T) -> bool,
    {
        pattern::loop_next::<true, _>(|| self.next_back(hs, pred))
    }

    fn next_reject_back<'hs, T, F>(&mut self, hs: &'hs [T], pred: &mut F) -> Option<(usize, usize)>
    where F: FnMut(&'hs T) -> bool,
    {
        if self.start >= self.end {
            return None;
        }

        if self.is_match_fw {
            self.end -= 1;
        }
        self.end -= self.count_back(true, hs, pred);

        let count = self.count_back(false, hs, pred);
        if count == 0 {
            None
        } else {
            self.is_match_bw = true;
            let pos = self.end;
            self.end -= count;
            Some((self.end, pos))
        }
    }

    fn count<'hs, T, F>(&self, want: bool, hs: &'hs [T], pred: &mut F) -> usize
    where F: FnMut(&'hs T) -> bool,
    {
        hs[self.start..self.end]
            .iter()
            .map(pred)
            .take_while(|&matches| matches == want)
            .count()
    }

    fn count_back<'hs, T, F>(&self, want: bool, hs: &'hs [T], pred: &mut F) -> usize
    where F: FnMut(&'hs T) -> bool,
    {
        hs[self.start..self.end]
            .iter()
            .rev()
            .map(pred)
            .take_while(|&matches| matches == want)
            .count()
    }
}

/////////////////////////////////////////////////////////////////////////////
// Searching for a subslice element
/////////////////////////////////////////////////////////////////////////////

// TODO: Implement something smarter perhaps?  Or have specialisation for
// different T?  We’re not using core::str::pattern::TwoWaySearcher because it
// requires PartialOrd elements.  Specifically, TwoWaySearcher::maximal_suffix
// and TwoWaySearcher::reverse_maximal_suffix methods compare elements.  For the
// time being, use naive O(nk) search.
#[derive(Clone, Debug)]
pub(super) struct NaiveSearcherState {
    start: usize,
    end: usize,
    is_match_fw: bool,
    is_match_bw: bool,
}

impl NaiveSearcherState {
    pub(super) fn new(haystack_length: usize) -> Self {
        Self {
            start: 0,
            end: haystack_length,
            is_match_fw: false,
            is_match_bw: false,
        }
    }

    pub(super) fn next<T: PartialEq>(&mut self, haystack: &[T], needle: &[T]) -> SearchStep {
        if self.end - self.start < needle.len() {
            SearchStep::Done
        } else if self.is_match_fw {
            let pos = self.start;
            self.start += needle.len();
            self.is_match_fw = false;
            SearchStep::Match(pos, self.start)
        } else {
            let count = haystack[self.start..self.end]
                .windows(needle.len())
                .take_while(|window| *window != needle)
                .count();
            let pos = self.start;
            if count == 0 {
                self.start += needle.len();
                SearchStep::Match(pos, self.start)
            } else {
                let pos = self.start;
                self.start += count;
                // We’ve either reached the end of the haystack or start
                // where it matches so maker is_match_fw.
                self.is_match_fw = true;
                SearchStep::Reject(pos, self.start)
            }
        }
    }

    pub(super) fn next_back<T: PartialEq>(&mut self, haystack: &[T], needle: &[T]) -> SearchStep {
        if self.end - self.start < needle.len() {
            SearchStep::Done
        } else if self.is_match_bw {
            let pos = self.end;
            self.end -= needle.len();
            self.is_match_bw = false;
            SearchStep::Match(self.end, pos)
        } else {
            let count = haystack[self.start..self.end]
                .windows(needle.len())
                .rev()
                .take_while(|window| *window != needle)
                .count();
            let pos = self.end;
            if count == 0 {
                self.end -= needle.len();
                SearchStep::Match(self.end, pos)
            } else {
                self.end -= count;
                // We’ve either reached the end of the haystack or start
                // where it matches so maker is_match_bw.
                self.is_match_bw = true;
                SearchStep::Reject(self.end, pos)
            }
        }
    }

    pub(super) fn next_match<T: PartialEq>(&mut self, haystack: &[T], needle: &[T]) -> Option<(usize, usize)> {
        pattern::loop_next::<true, _>(|| self.next(haystack, needle))
    }

    pub(super) fn next_reject<T: PartialEq>(&mut self, haystack: &[T], needle: &[T]) -> Option<(usize, usize)> {
        pattern::loop_next::<false, _>(|| self.next(haystack, needle))
    }

    pub(super) fn next_match_back<T: PartialEq>(&mut self, haystack: &[T], needle: &[T]) -> Option<(usize, usize)> {
        pattern::loop_next::<true, _>(|| self.next_back(haystack, needle))
    }

    pub(super) fn next_reject_back<T: PartialEq>(&mut self, haystack: &[T], needle: &[T]) -> Option<(usize, usize)> {
        pattern::loop_next::<false, _>(|| self.next_back(haystack, needle))
    }
}
