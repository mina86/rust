//! The Pattern API.
//!
//! The Pattern API provides a generic mechanism for using different pattern
//! types when searching through different objects.
//!
//! For more details, see the traits [`Pattern`], [`Haystack`], [`Searcher`],
//! [`ReverseSearcher`] and [`DoubleEndedSearcher`].  Although this API is
//! unstable, it is exposed via stable methods on corresponding haystack types.
//!
//! # Examples
//!
//! [`Pattern<&str>`] is [implemented][pattern-impls] in the stable API for
//! [`&str`][`str`], [`char`], slices of [`char`], and functions and closures
//! implementing `FnMut(char) -> bool`.
//!
//! ```
//! let s = "Can you find a needle in a haystack?";
//!
//! // &str pattern
//! assert_eq!(s.find("you"), Some(4));
//! // char pattern
//! assert_eq!(s.find('n'), Some(2));
//! // array of chars pattern
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u']), Some(1));
//! // slice of chars pattern
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u'][..]), Some(1));
//! // closure pattern
//! assert_eq!(s.find(|c: char| c.is_ascii_punctuation()), Some(35));
//! ```
//!
//! [pattern-impls]: Pattern#implementors

#![unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]

use crate::fmt;
use crate::mem::{replace, take};
use crate::ops::Range;

/// A pattern which can be matched against a [`Haystack`].
///
/// A `Pattern<H>` expresses that the implementing type can be used as a pattern
/// for searching in an `H`.  For example, character `'a'` and string `"aa"` are
/// patterns that would match at index `1` in the string `"baaaab"`.
///
/// The trait itself acts as a builder for an associated [`Searcher`] type,
/// which does the actual work of finding occurrences of the pattern in
/// a string.
///
/// Depending on the type of the haystack and the pattern, the semantics of the
/// pattern can change.  The table below describes some of those behaviours for
/// a [`&str`][str] haystack.
///
/// | Pattern type             | Match condition                           |
/// |--------------------------|-------------------------------------------|
/// | `&str`                   | is substring                              |
/// | `char`                   | is contained in string                    |
/// | `&[char]`                | any char in slice is contained in string  |
/// | `F: FnMut(char) -> bool` | `F` returns `true` for a char in string   |
///
/// # Examples
///
/// ```
/// // &str pattern matching &str
/// assert_eq!("abaaa".find("ba"), Some(1));
/// assert_eq!("abaaa".find("bac"), None);
///
/// // char pattern matching &str
/// assert_eq!("abaaa".find('a'), Some(0));
/// assert_eq!("abaaa".find('b'), Some(1));
/// assert_eq!("abaaa".find('c'), None);
///
/// // &[char; N] pattern matching &str
/// assert_eq!("ab".find(&['b', 'a']), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z']), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd']), None);
///
/// // &[char] pattern matching &str
/// assert_eq!("ab".find(&['b', 'a'][..]), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z'][..]), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd'][..]), None);
///
/// // FnMut(char) -> bool pattern matching &str
/// assert_eq!("abcdef_z".find(|ch| ch > 'd' && ch < 'y'), Some(4));
/// assert_eq!("abcddd_z".find(|ch| ch > 'd' && ch < 'y'), None);
/// ```
pub trait Pattern<H: Haystack>: Sized {
    /// Associated searcher for this pattern.
    type Searcher: Searcher<H>;

    /// Constructs the associated searcher from `self` and the `haystack` to
    /// search in.
    fn into_searcher(self, haystack: H) -> Self::Searcher;

    /// Checks whether the pattern matches anywhere in the haystack.
    fn is_contained_in(self, haystack: H) -> bool {
        self.into_searcher(haystack).next_match().is_some()
    }

    /// Checks whether the pattern matches at the front of the haystack.
    fn is_prefix_of(self, haystack: H) -> bool {
        matches!(self.into_searcher(haystack).next(), SearchStep::Match(..))
    }

    /// Checks whether the pattern matches at the back of the haystack.
    fn is_suffix_of(self, haystack: H) -> bool
    where
        Self::Searcher: ReverseSearcher<H>,
    {
        matches!(self.into_searcher(haystack).next_back(), SearchStep::Match(..))
    }

    /// Removes the pattern from the front of haystack, if it matches.
    fn strip_prefix_of(self, haystack: H) -> Option<H> {
        if let SearchStep::Match(start, pos) = self.into_searcher(haystack).next() {
            // This cannot be debug_assert_eq because StartCursor isn’t Debug.
            debug_assert!(
                start == haystack.cursor_at_front(),
                "The first search step from Searcher \
                 must include the first character"
            );
            let end = haystack.cursor_at_back();
            // SAFETY: `Searcher` is known to return valid indices.
            Some(unsafe { haystack.get_unchecked(pos..end) })
        } else {
            None
        }
    }

    /// Removes the pattern from the back of haystack, if it matches.
    fn strip_suffix_of(self, haystack: H) -> Option<H>
    where
        Self::Searcher: ReverseSearcher<H>,
    {
        if let SearchStep::Match(pos, end) = self.into_searcher(haystack).next_back() {
            // This cannot be debug_assert_eq because StartCursor isn’t Debug.
            debug_assert!(
                end == haystack.cursor_at_back(),
                "The first search step from ReverseSearcher \
                 must include the last character"
            );
            let start = haystack.cursor_at_front();
            // SAFETY: `Searcher` is known to return valid indices.
            Some(unsafe { haystack.get_unchecked(start..pos) })
        } else {
            None
        }
    }
}

/// A type which can be searched in using a [`Pattern`].
///
/// The trait is used in combination with [`Pattern`] trait to express a pattern
/// that can be used to search for elements in given haystack.
pub trait Haystack: Sized + Copy {
    /// A cursor representing position in the haystack or its end.
    type Cursor: Copy + PartialEq;

    /// Returns cursor pointing at the beginning of the haystack.
    fn cursor_at_front(self) -> Self::Cursor;

    /// Returns cursor pointing at the end of the haystack.
    fn cursor_at_back(self) -> Self::Cursor;

    /// Returns whether the haystack is empty.
    fn is_empty(self) -> bool;

    /// Returns portions of the haystack indicated by the cursor range.
    ///
    /// # Safety
    ///
    /// Range’s start and end must be valid haystack split positions.
    /// Furthermore, start mustn’t point at position after end.
    ///
    /// A valid split positions are:
    /// - the front of the haystack (as returned by
    ///   [`cursor_at_front()`][Self::cursor_at_front],
    /// - the back of the haystack (as returned by
    ///   [`cursor_at_back()`][Self::cursor_at_back] or
    /// - any cursor returned by a [`Searcher`] or [`ReverseSearcher`].
    unsafe fn get_unchecked(self, range: Range<Self::Cursor>) -> Self;
}

/// Result of calling [`Searcher::next()`] or [`ReverseSearcher::next_back()`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SearchStep<T = usize> {
    /// Expresses that a match of the pattern has been found at
    /// `haystack[a..b]`.
    Match(T, T),
    /// Expresses that `haystack[a..b]` has been rejected as a possible match of
    /// the pattern.
    ///
    /// Note that there might be more than one `Reject` between two `Match`es,
    /// there is no requirement for them to be combined into one.
    Reject(T, T),
    /// Expresses that every element of the haystack has been visited, ending
    /// the iteration.
    Done,
}

/// Possible return type of a search.
///
/// It abstract differences between `next`, `next_match` and `next_reject`
/// methods.  Depending on return type an implementation for those functions
/// will generate matches and rejects, only matches or only rejects.
#[unstable(feature = "pattern_internals", issue = "none")]
pub trait SearchResult<T = usize>: Sized + sealed::Sealed {
    /// Value indicating searching has finished.
    const DONE: Self;

    /// Whether search should return reject as soon as possible.
    ///
    /// For example, if a search can quickly determine that the very next
    /// position cannot be where a next match starts, it should return a reject
    /// with that position.  This is an optimisation which allows the algorithm
    /// to not waste time looking for the next match if caller is only
    /// interested in the next position of a reject.
    ///
    /// If this is `true`, [`rejecting()`][Self::rejecting] is guaranteed to
    /// return `Some` and if this is `false`, [`matching()`][Self::matching] is
    /// guaranteed to return `Some`.
    const USE_EARLY_REJECT: bool;

    /// Returns value describing a match or `None` if this implementation
    /// doesn’t care about matches.
    fn matching(start: T, end: T) -> Option<Self>;

    /// Returns value describing a reject or `None` if this implementation
    /// doesn’t care about matches.
    fn rejecting(start: T, end: T) -> Option<Self>;
}

/// A wrapper for result type which only carries information about matches.
#[unstable(feature = "pattern_internals", issue = "none")]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct MatchOnly<T = usize>(pub Option<(T, T)>);

/// A wrapper for result type which only carries information about rejects.
#[unstable(feature = "pattern_internals", issue = "none")]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct RejectOnly<T = usize>(pub Option<(T, T)>);

impl<T> SearchResult<T> for SearchStep<T> {
    const DONE: Self = SearchStep::Done;
    const USE_EARLY_REJECT: bool = false;

    #[inline(always)]
    fn matching(s: T, e: T) -> Option<Self> {
        Some(SearchStep::Match(s, e))
    }

    #[inline(always)]
    fn rejecting(s: T, e: T) -> Option<Self> {
        Some(SearchStep::Reject(s, e))
    }
}

impl<T> SearchResult<T> for MatchOnly<T> {
    const DONE: Self = Self(None);
    const USE_EARLY_REJECT: bool = false;

    #[inline(always)]
    fn matching(s: T, e: T) -> Option<Self> {
        Some(Self(Some((s, e))))
    }

    #[inline(always)]
    fn rejecting(_s: T, _e: T) -> Option<Self> {
        None
    }
}

impl<T> From<MatchOnly<T>> for Option<(T, T)> {
    fn from(m: MatchOnly<T>) -> Self {
        m.0
    }
}

impl<T> SearchResult<T> for RejectOnly<T> {
    const DONE: Self = Self(None);
    const USE_EARLY_REJECT: bool = true;

    #[inline(always)]
    fn matching(_s: T, _e: T) -> Option<Self> {
        None
    }

    #[inline(always)]
    fn rejecting(s: T, e: T) -> Option<Self> {
        Some(Self(Some((s, e))))
    }
}

impl<T> From<RejectOnly<T>> for Option<(T, T)> {
    fn from(m: RejectOnly<T>) -> Self {
        m.0
    }
}

mod sealed {
    pub trait Sealed {}
    impl<T> Sealed for super::SearchStep<T> {}
    impl<T> Sealed for super::MatchOnly<T> {}
    impl<T> Sealed for super::RejectOnly<T> {}
}

/// A searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping matches of
/// a pattern starting from the front of a haystack `H`.
///
/// It will be implemented by associated `Searcher` types of the [`Pattern`]
/// trait.
///
/// The trait is marked unsafe because the indices returned by the
/// [`next()`][Searcher::next] methods are required to lie on valid haystack
/// split positions.  This enables consumers of this trait to slice the haystack
/// without additional runtime checks.
pub unsafe trait Searcher<H: Haystack> {
    /// Getter for the underlying string to be searched in
    ///
    /// Will always return the same haystack that was used when creating the
    /// searcher.
    fn haystack(&self) -> H;

    /// Performs the next search step starting from the front.
    ///
    /// - Returns [`Match(a, b)`][SearchStep::Match] if `haystack[a..b]` matches
    ///   the pattern.
    /// - Returns [`Reject(a, b)`][SearchStep::Reject] if `haystack[a..b]` can
    ///   not match the pattern, even partially.
    /// - Returns [`Done`][SearchStep::Done] if every byte of the haystack has
    ///   been visited.
    ///
    /// The stream of [`Match`][SearchStep::Match] and
    /// [`Reject`][SearchStep::Reject] values up to a [`Done`][SearchStep::Done]
    /// will contain index ranges that are adjacent, non-overlapping,
    /// covering the whole haystack, and laying on utf8 boundaries.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero length.
    ///
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"` might
    /// produce the stream `[Reject(0, 1), Reject(1, 2), Match(2, 5), Reject(5,
    /// 8)]`
    fn next(&mut self) -> SearchStep<H::Cursor>;

    /// Finds the next [`Match`][SearchStep::Match] result. See
    /// [`next()`][Searcher::next].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the
    /// returned ranges of this and [`next_reject`][Searcher::next_reject] will
    /// overlap.  This will return `(start_match, end_match)`, where start_match
    /// is the index of where the match begins, and end_match is the index after
    /// the end of the match.
    fn next_match(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop {
            match self.next() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result.  See
    /// [`next()`][Searcher::next] and [`next_match()`][Searcher::next_match].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the
    /// returned ranges of this and [`next_match`][Searcher::next_match] will
    /// overlap.
    fn next_reject(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop {
            match self.next() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

/// A reverse searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping matches of
/// a pattern starting from the back of a haystack `H`.
///
/// It will be implemented by associated [`Searcher`] types of the [`Pattern`]
/// trait if the pattern supports searching for it from the back.
///
/// The index ranges returned by this trait are not required to exactly match
/// those of the forward search in reverse.
///
/// For the reason why this trait is marked unsafe, see the parent trait
/// [`Searcher`].
pub unsafe trait ReverseSearcher<H: Haystack>: Searcher<H> {
    /// Performs the next search step starting from the back.
    ///
    /// - Returns [`Match(a, b)`][SearchStep::Match] if `haystack[a..b]`
    ///   matches the pattern.
    /// - Returns [`Reject(a, b)`][SearchStep::Reject] if `haystack[a..b]`
    ///   can not match the pattern, even partially.
    /// - Returns [`Done`][SearchStep::Done] if every byte of the haystack
    ///   has been visited
    ///
    /// The stream of [`Match`][SearchStep::Match] and
    /// [`Reject`][SearchStep::Reject] values up to a [`Done`][SearchStep::Done]
    /// will contain index ranges that are adjacent, non-overlapping, covering
    /// the whole haystack, and laying on utf8 boundaries.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero
    /// length.
    ///
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"` might
    /// produce the stream `[Reject(7, 8), Match(4, 7), Reject(1, 4), Reject(0,
    /// 1)]`.
    fn next_back(&mut self) -> SearchStep<H::Cursor>;

    /// Finds the next [`Match`][SearchStep::Match] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    fn next_match_back(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop {
            match self.next_back() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    fn next_reject_back(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop {
            match self.next_back() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

/// A marker trait to express that a [`ReverseSearcher`] can be used for
/// a [`DoubleEndedIterator`] implementation.
///
/// For this, the impl of [`Searcher`] and [`ReverseSearcher`] need to follow
/// these conditions:
///
/// - All results of `next()` need to be identical to the results of
///   `next_back()` in reverse order.
/// - `next()` and `next_back()` need to behave as the two ends of a range of
///   values, that is they can not "walk past each other".
///
/// # Examples
///
/// `char::Searcher` is a `DoubleEndedSearcher` because searching for a [`char`]
/// only requires looking at one at a time, which behaves the same from both
/// ends.
///
/// `(&str)::Searcher` is not a `DoubleEndedSearcher` because the pattern `"aa"`
/// in the haystack `"aaa"` matches as either `"[aa]a"` or `"a[aa]"`, depending
/// from which side it is searched.
pub trait DoubleEndedSearcher<H: Haystack>: ReverseSearcher<H> {}

/// A wrapper around single-argument function returning a boolean,
/// i.e. a predicate function.
///
/// `Predicate` objects are created with [`predicate`] function.
#[derive(Clone, Debug)]
pub struct Predicate<F>(F);

/// Constructs a wrapper for a single-argument function returning a boolean,
/// i.e. a predicate function.
///
/// This is intended to be used as a pattern when working with haystacks which
/// (for whatever reason) cannot support naked function traits as patterns.
///
/// # Examples
///
/// ```
/// # #![feature(pattern, slice_pattern)]
/// use core::pattern::predicate;
///
/// let nums = &[10, 40, 30, 40];
/// assert_eq!(nums.find(predicate(|n| n % 3 == 0)), Some(2));
/// assert_eq!(nums.find(predicate(|n| n % 2 == 1)), None);
/// ```
pub fn predicate<T, F: FnMut(T) -> bool>(pred: F) -> Predicate<F> {
    Predicate(pred)
}

impl<F> Predicate<F> {
    /// Executes the predicate returning its result.
    pub fn test<T>(&mut self, element: T) -> bool
    where
        F: FnMut(T) -> bool,
    {
        self.0(element)
    }

    /// Returns reference to the wrapped predicate function.
    pub fn as_fn<T>(&mut self) -> &mut F
    where
        F: FnMut(T) -> bool,
    {
        &mut self.0
    }

    /// Consumes this object and returns wrapped predicate function.
    pub fn into_fn<T>(self) -> F
    where
        F: FnMut(T) -> bool,
    {
        self.0
    }
}

//////////////////////////////////////////////////////////////////////////////
// Internal EmptyNeedleSearcher helper
//////////////////////////////////////////////////////////////////////////////

/// Helper for implementing searchers looking for empty patterns.
///
/// An empty pattern matches around every element of a haystack.  For example,
/// within a `&str` it matches around every character.  (This includes at the
/// beginning and end of the string).
///
/// This struct helps implement searchers for empty patterns for various
/// haystacks.   The only requirement is a function which advances the start
/// position or end position of the haystack range.
///
/// # Examples
///
/// ```
/// #![feature(pattern, pattern_internals)]
/// use core::pattern::{EmptyNeedleSearcher, SearchStep};
///
/// let haystack = "fóó";
/// let mut searcher = EmptyNeedleSearcher::new(haystack);
/// let advance = |range: core::ops::Range<usize>| {
///     range.start + haystack[range].chars().next().unwrap().len_utf8()
/// };
/// let steps = core::iter::from_fn(|| {
///     match searcher.next_fwd(advance) {
///         SearchStep::Done => None,
///         step => Some(step)
///     }
/// }).collect::<Vec<_>>();
/// assert_eq!(&[
///     SearchStep::Match(0, 0),
///     SearchStep::Reject(0, 1),
///     SearchStep::Match(1, 1),
///     SearchStep::Reject(1, 3),
///     SearchStep::Match(3, 3),
///     SearchStep::Reject(3, 5),
///     SearchStep::Match(5, 5),
/// ], steps.as_slice());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[unstable(feature = "pattern_internals", issue = "none")]
pub struct EmptyNeedleSearcher<T> {
    start: T,
    end: T,
    is_match_fwd: bool,
    is_match_bwd: bool,
    // Needed in case of an empty haystack, see #85462
    is_finished: bool,
}

impl<T: Copy + PartialOrd> EmptyNeedleSearcher<T> {
    /// Creates a new empty needle searcher for given haystack.
    ///
    /// The haystack is used to initialise the range of valid cursors positions.
    pub fn new<H: Haystack<Cursor = T>>(haystack: H) -> Self {
        Self {
            start: haystack.cursor_at_front(),
            end: haystack.cursor_at_back(),
            is_match_bwd: true,
            is_match_fwd: true,
            is_finished: false,
        }
    }

    /// Returns next search result.
    ///
    /// The callback function is used to advance the **start** of the range the
    /// searcher is working on.  It is passed the current range of cursor
    /// positions that weren’t visited yet and it must return the new start
    /// cursor position.  It’s never called with an empty range.  For some
    /// haystacks the callback may be as simple as a closure returning the start
    /// incremented by one; others might require looking for a new valid
    /// boundary.
    pub fn next_fwd<R: SearchResult<T>, F>(&mut self, advance_fwd: F) -> R
    where
        F: FnOnce(crate::ops::Range<T>) -> T,
    {
        if self.is_finished {
            return R::DONE;
        }
        if take(&mut self.is_match_fwd) {
            if let Some(ret) = R::matching(self.start, self.start) {
                return ret;
            }
        }
        if self.start < self.end {
            let pos = self.start;
            self.start = advance_fwd(self.start..self.end);
            if let Some(ret) = R::rejecting(pos, self.start) {
                self.is_match_fwd = true;
                return ret;
            }
            return R::matching(self.start, self.start).unwrap();
        }
        self.is_finished = true;
        R::DONE
    }

    /// Returns next search result.
    ///
    /// The callback function is used to advance the **end** of the range the
    /// searcher is working on backwards.  It is passed the current range of
    /// cursor positions that weren’t visited yet and it must return the new end
    /// cursor position.  It’s never called with an empty range.  For some
    /// haystacks the callback may be as simple as a closure returning the end
    /// decremented by one; others might require looking for a new valid
    /// boundary.
    pub fn next_bwd<R: SearchResult<T>, F>(&mut self, advance_bwd: F) -> R
    where
        F: FnOnce(crate::ops::Range<T>) -> T,
    {
        if self.is_finished {
            return R::DONE;
        }
        if take(&mut self.is_match_bwd) {
            if let Some(ret) = R::matching(self.end, self.end) {
                return ret;
            }
        }
        if self.start < self.end {
            let pos = self.end;
            self.end = advance_bwd(self.start..self.end);
            if let Some(ret) = R::rejecting(self.end, pos) {
                self.is_match_bwd = true;
                return ret;
            }
            return R::matching(self.end, self.end).unwrap();
        }
        self.is_finished = true;
        R::DONE
    }
}

//////////////////////////////////////////////////////////////////////////////
// Internal Split and SplitN implementations
//////////////////////////////////////////////////////////////////////////////

/// Helper type for implementing split iterators.
///
/// It’s a generic type which works with any [`Haystack`] and [`Searcher`] over
/// that haystack.  Intended usage is to create a newtype wrapping this type
/// which implements iterator interface on top of [`next_fwd`][Split::next_fwd]
/// or [`next_fwd`][Split::next_fwd] methods.
///
/// Note that unless `S` implements [`DoubleEndedSearcher`] trait, it’s
/// incorrect to use this type to implement a double ended iterator.
///
/// For an example of this type in use, see [`core::str::Split`].
#[unstable(feature = "pattern_internals", issue = "none")]
pub struct Split<H: Haystack, S: Searcher<H>> {
    /// Start of the region of the haystack yet to be examined.
    start: H::Cursor,
    /// End of the region of the haystack yet to be examined.
    end: H::Cursor,
    /// Searcher returning matches of the delimiter pattern.
    searcher: S,
    /// Whether to return an empty part if there’s delimiter at the end of the
    /// haystack.
    allow_trailing_empty: bool,
    /// Whether splitting has finished.
    finished: bool,
}

/// Helper type for implementing split iterators with a split limit.
///
/// It’s like [`Split`] but limits number of parts the haystack will be split
/// into.
#[unstable(feature = "pattern_internals", issue = "none")]
pub struct SplitN<H: Haystack, S: Searcher<H>> {
    /// Inner split implementation.
    inner: Split<H, S>,
    /// Maximum number of parts the haystack can be split into.
    limit: usize,
}

impl<H: Haystack, S: Searcher<H> + Clone> Clone for Split<H, S> {
    fn clone(&self) -> Self {
        Self { searcher: self.searcher.clone(), ..*self }
    }
}

impl<H: Haystack, S: Searcher<H> + Clone> Clone for SplitN<H, S> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), ..*self }
    }
}

impl<H, S> fmt::Debug for Split<H, S>
where
    H: Haystack<Cursor: fmt::Debug>,
    S: Searcher<H> + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Split")
            .field("start", &self.start)
            .field("end", &self.end)
            .field("searcher", &self.searcher)
            .field("allow_trailing_empty", &self.allow_trailing_empty)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<H, S> fmt::Debug for SplitN<H, S>
where
    H: Haystack<Cursor: fmt::Debug>,
    S: Searcher<H> + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("SplitN").field("inner", &self.inner).field("limit", &self.limit).finish()
    }
}

impl<H: Haystack, S: Searcher<H>> Split<H, S> {
    /// Creates a new object configured without a limit and with
    /// `allow_trailing_empty` option disabled.
    ///
    /// To set `allow_trailing_empty`, use
    /// [`with_allow_trailing_empty()`][Self::with_allow_trailing_empty] method.
    /// To set split limit, use [`with_limit()`][Self::with_limit] method.
    pub fn new(searcher: S) -> Self {
        let haystack = searcher.haystack();
        Self {
            searcher,
            start: haystack.cursor_at_front(),
            end: haystack.cursor_at_back(),
            allow_trailing_empty: false,
            finished: false,
        }
    }

    /// Changes splits limit from unlimited to given value.
    ///
    /// The limit specifies maximum number of parts haystack will be split into.
    pub fn with_limit(self, limit: usize) -> SplitN<H, S> {
        SplitN { inner: self, limit }
    }

    /// Enables allow_trailing_empty option.
    ///
    /// If enabled (which is not the default), if the haystack is empty or
    /// terminated by a pattern match, the last haystack part returned will be
    /// empty.  Otherwise, the last empty split is not returned.
    pub fn with_allow_trailing_empty(mut self) -> Self {
        self.allow_trailing_empty = true;
        self
    }
}

impl<H: Haystack, S: Searcher<H>> Split<H, S> {
    /// Returns next part of the haystack or `None` if splitting is done.
    ///
    /// If `INCLUSIVE` is `true`, returned value will include the matching
    /// pattern.
    pub fn next_fwd<const INCLUSIVE: bool>(&mut self) -> Option<H> {
        if self.finished {
            return None;
        }
        let haystack = self.searcher.haystack();
        if let Some((start, end)) = self.searcher.next_match() {
            let range = self.start..(if INCLUSIVE { end } else { start });
            self.start = end;
            // SAFETY: self.start and self.end come from Haystack or Searcher
            // and thus are guaranteed to be valid split positions.
            Some(unsafe { haystack.get_unchecked(range) })
        } else {
            self.get_end()
        }
    }

    /// Returns next looking from back of the haystack part of the haystack or
    /// `None` if splitting is done.
    ///
    /// If `INCLUSIVE` is `true`, returned value will include the matching
    /// pattern.
    pub fn next_bwd<const INCLUSIVE: bool>(&mut self) -> Option<H>
    where
        S: ReverseSearcher<H>,
    {
        if self.finished {
            return None;
        }

        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            if let Some(elt) = self.next_bwd::<INCLUSIVE>() {
                if !elt.is_empty() {
                    return Some(elt);
                }
            }
            if self.finished {
                return None;
            }
        }

        let range = if let Some((start, end)) = self.searcher.next_match_back() {
            end..replace(&mut self.end, if INCLUSIVE { end } else { start })
        } else {
            self.finished = true;
            self.start..self.end
        };
        // SAFETY: All indices come from Haystack or Searcher which guarantee
        // that they are valid split positions.
        Some(unsafe { self.searcher.haystack().get_unchecked(range) })
    }

    /// Returns remaining part of the haystack that hasn’t been processed yet.
    pub fn remainder(&self) -> Option<H> {
        (!self.finished).then(|| {
            // SAFETY: self.start and self.end come from Haystack or Searcher
            // and thus are guaranteed to be valid split positions.
            unsafe { self.searcher.haystack().get_unchecked(self.start..self.end) }
        })
    }

    /// Returns the final haystack part.
    ///
    /// Sets `finished` flag so any further calls to this or other methods will
    /// return `None`.
    fn get_end(&mut self) -> Option<H> {
        if !self.finished {
            self.finished = true;
            if self.allow_trailing_empty || self.start != self.end {
                // SAFETY: self.start and self.end come from Haystack or
                // Searcher and thus are guaranteed to be valid split positions.
                return Some(unsafe {
                    self.searcher.haystack().get_unchecked(self.start..self.end)
                });
            }
        }
        None
    }
}

impl<H: Haystack, S: Searcher<H>> SplitN<H, S> {
    /// Returns next part of the haystack or `None` if splitting is done.
    ///
    /// If `INCLUSIVE` is `true`, returned value will include the matching
    /// pattern.
    pub fn next_fwd<const INCLUSIVE: bool>(&mut self) -> Option<H> {
        match self.dec_limit()? {
            0 => self.inner.get_end(),
            _ => self.inner.next_fwd::<INCLUSIVE>(),
        }
    }

    /// Returns next looking from back of the haystack part of the haystack or
    /// `None` if splitting is done.
    ///
    /// If `INCLUSIVE` is `true`, returned value will include the matching
    /// pattern.
    pub fn next_bwd<const INCLUSIVE: bool>(&mut self) -> Option<H>
    where
        S: ReverseSearcher<H>,
    {
        match self.dec_limit()? {
            0 => self.inner.get_end(),
            _ => self.inner.next_bwd::<INCLUSIVE>(),
        }
    }

    /// Returns remaining part of the haystack that hasn’t been processed yet.
    pub fn remainder(&self) -> Option<H> {
        self.inner.remainder()
    }

    /// Decrements limit and returns its new value or None if it’s already zero.
    fn dec_limit(&mut self) -> Option<usize> {
        self.limit = self.limit.checked_sub(1)?;
        Some(self.limit)
    }
}
