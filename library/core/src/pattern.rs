//! The Pattern API.
//!
//! The Pattern API provides a generic mechanism for using different pattern
//! types when searching through different objects.
//!
//! For more details, see the traits [`Pattern`], [`Haystack`], [`Searcher`],
//! [`ReverseSearcher`] and [`DoubleEndedSearcher`].  Although this API is
//! unstable, it is exposed via stable APIs on the [`str`] type.
//!
//! # Examples
//!
//! [`Pattern`] is [implemented][pattern-impls] in the stable API for
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

use crate::marker::PhantomData;

/// A pattern which can be matched against a [`Haystack`].
///
/// A `Pattern<H>` expresses that the implementing type can be used as a pattern
/// for searching in a `H`.
///
/// For example, character `'a'` and string `"aa"` are patterns that would match
/// at index `1` in the string `"baaaab"`.
///
/// The trait itself acts as a builder for an associated
/// [`Searcher`] type, which does the actual work of finding
/// occurrences of the pattern in a string.
///
/// Depending on the type of the pattern, the behaviour of methods like
/// [`str::find`] and [`str::contains`] can change. The table below describes
/// some of those behaviours.
///
/// | Pattern type             | Match condition                           |
/// |--------------------------|-------------------------------------------|
/// | `&str`                   | is substring                              |
/// | `char`                   | is contained in string                    |
/// | `&[char]`                | any char in slice is contained in string  |
/// | `F: FnMut(char) -> bool` | `F` returns `true` for a char in string   |
/// | `&&str`                  | is substring                              |
/// | `&String`                | is substring                              |
///
/// # Examples
///
/// ```
/// // &str
/// assert_eq!("abaaa".find("ba"), Some(1));
/// assert_eq!("abaaa".find("bac"), None);
///
/// // char
/// assert_eq!("abaaa".find('a'), Some(0));
/// assert_eq!("abaaa".find('b'), Some(1));
/// assert_eq!("abaaa".find('c'), None);
///
/// // &[char; N]
/// assert_eq!("ab".find(&['b', 'a']), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z']), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd']), None);
///
/// // &[char]
/// assert_eq!("ab".find(&['b', 'a'][..]), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z'][..]), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd'][..]), None);
///
/// // FnMut(char) -> bool
/// assert_eq!("abcdef_z".find(|ch| ch > 'd' && ch < 'y'), Some(4));
/// assert_eq!("abcddd_z".find(|ch| ch > 'd' && ch < 'y'), None);
/// ```
pub trait Pattern<H: Haystack>: Sized {
    /// Associated searcher for this pattern
    type Searcher: Searcher<H>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: H) -> Self::Searcher;

    /// Checks whether the pattern matches anywhere in the haystack
    fn is_contained_in(self, haystack: H) -> bool {
        self.into_searcher(haystack).next_match().is_some()
    }

    /// Checks whether the pattern matches at the front of the haystack
    fn is_prefix_of(self, haystack: H) -> bool {
        matches!(
            self.into_searcher(haystack).next(),
            SearchStep::Match(start, _) if start == haystack.cursor_at_front()
        )
    }

    /// Checks whether the pattern matches at the back of the haystack
    fn is_suffix_of(self, haystack: H) -> bool
    where Self::Searcher: ReverseSearcher<H> {
        matches!(
            self.into_searcher(haystack).next_back(),
            SearchStep::Match(_, end) if end == haystack.cursor_at_back()
        )
    }

    /// Removes the pattern from the front of haystack, if it matches.
    fn strip_prefix_of(self, haystack: H) -> Option<H> {
        if let SearchStep::Match(start, end) = self.into_searcher(haystack).next() {
            // This cannot be debug_assert_eq because StartCursor isn’t Debug.
            debug_assert!(start == haystack.cursor_at_front(),
                          "The first search step from Searcher \
                           must include the first character");
            // SAFETY: `Searcher` is known to return valid indices.
            Some(unsafe { haystack.split_at_cursor_unchecked(end) }.1)
        } else {
            None
        }
    }

    /// Removes the pattern from the back of haystack, if it matches.
    fn strip_suffix_of(self, haystack: H) -> Option<H>
    where Self::Searcher: ReverseSearcher<H> {
        if let SearchStep::Match(start, end) = self.into_searcher(haystack).next_back() {
            // This cannot be debug_assert_eq because StartCursor isn’t Debug.
            debug_assert!(end == haystack.cursor_at_back(),
                          "The first search step from ReverseSearcher \
                           must include the last character");
            // SAFETY: `Searcher` is known to return valid indices.
            Some(unsafe { haystack.split_at_cursor_unchecked(start) }.0)
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
    type Cursor: Copy + PartialOrd;

    /// Returns cursor pointing at the beginning of the haystack.
    fn cursor_at_front(&self) -> Self::Cursor;

    /// Returns cursor pointing at the end of the haystack.
    fn cursor_at_back(&self) -> Self::Cursor;

    /// Splits haystack into two at given cursor position.
    ///
    /// Note that splitting a haystack isn’t guaranteed to preserve total
    /// length.  That is, each separate part’s length may be longer than length
    /// of the original haystack.  This property is preserved for `&str` and
    /// `&[T]` haystacks but not for `&OsStr`.
    unsafe fn split_at_cursor_unchecked(self, cursor: Self::Cursor) -> (Self, Self);
}


/// Result of calling [`Searcher::next()`] or [`ReverseSearcher::next_back()`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SearchStep<T = usize> {
    /// Expresses that a match of the pattern has been found at
    /// `haystack[a..b]`.
    Match(T, T),
    /// Expresses that `haystack[a..b]` has been rejected as a possible match
    /// of the pattern.
    ///
    /// Note that there might be more than one `Reject` between two `Match`es,
    /// there is no requirement for them to be combined into one.
    Reject(T, T),
    /// Expresses that every byte of the haystack has been visited, ending
    /// the iteration.
    Done,
}

/// A searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping
/// matches of a pattern starting from the front (left) of a string.
///
/// It will be implemented by associated `Searcher`
/// types of the [`Pattern`] trait.
///
/// The trait is marked unsafe because the indices returned by the
/// [`next()`][Searcher::next] methods are required to lie on valid utf8
/// boundaries in the haystack. This enables consumers of this trait to
/// slice the haystack without additional runtime checks.
pub unsafe trait Searcher<H: Haystack> {
    /// Getter for the underlying string to be searched in
    ///
    /// Will always return the same [`&str`][str].
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
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"`
    /// might produce the stream
    /// `[Reject(0, 1), Reject(1, 2), Match(2, 5), Reject(5, 8)]`
    fn next(&mut self) -> SearchStep<H::Cursor>;

    /// Finds the next [`Match`][SearchStep::Match] result. See [`next()`][Searcher::next].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the returned ranges
    /// of this and [`next_reject`][Searcher::next_reject] will overlap. This will return
    /// `(start_match, end_match)`, where start_match is the index of where
    /// the match begins, and end_match is the index after the end of the match.
    fn next_match(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop_next::<true, _>(|| self.next())
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result. See [`next()`][Searcher::next]
    /// and [`next_match()`][Searcher::next_match].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the returned ranges
    /// of this and [`next_match`][Searcher::next_match] will overlap.
    fn next_reject(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop_next::<false, _>(|| self.next())
    }
}

/// A reverse searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping
/// matches of a pattern starting from the back (right) of a string.
///
/// It will be implemented by associated [`Searcher`]
/// types of the [`Pattern`] trait if the pattern supports searching
/// for it from the back.
///
/// The index ranges returned by this trait are not required
/// to exactly match those of the forward search in reverse.
///
/// For the reason why this trait is marked unsafe, see the
/// parent trait [`Searcher`].
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
    /// will contain index ranges that are adjacent, non-overlapping,
    /// covering the whole haystack, and laying on utf8 boundaries.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero length.
    ///
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"`
    /// might produce the stream
    /// `[Reject(7, 8), Match(4, 7), Reject(1, 4), Reject(0, 1)]`.
    fn next_back(&mut self) -> SearchStep<H::Cursor>;

    /// Finds the next [`Match`][SearchStep::Match] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    fn next_match_back(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop_next::<true, _>(|| self.next_back())
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    fn next_reject_back(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop_next::<false, _>(|| self.next_back())
    }
}

/// A marker trait to express that a [`ReverseSearcher`]
/// can be used for a [`DoubleEndedIterator`] implementation.
///
/// For this, the impl of [`Searcher`] and [`ReverseSearcher`] need
/// to follow these conditions:
///
/// - All results of `next()` need to be identical
///   to the results of `next_back()` in reverse order.
/// - `next()` and `next_back()` need to behave as
///   the two ends of a range of values, that is they
///   can not "walk past each other".
///
/// # Examples
///
/// `char::Searcher` is a `DoubleEndedSearcher` because searching for a
/// [`char`] only requires looking at one at a time, which behaves the same
/// from both ends.
///
/// `(&str)::Searcher` is not a `DoubleEndedSearcher` because
/// the pattern `"aa"` in the haystack `"aaa"` matches as either
/// `"[aa]a"` or `"a[aa]"`, depending from which side it is searched.
pub trait DoubleEndedSearcher<H: Haystack>: ReverseSearcher<H> {}


/// XXX TODO placeholder
#[derive(Clone, Debug)]
pub struct Predicate<T, F>(F, PhantomData<*const T>);

/// XXX TODO placeholder
pub fn predicate<T, F: FnMut(T) -> bool>(pred: F) -> Predicate<T, F> {
    Predicate(pred, PhantomData)
}

impl<T, F: FnMut(T) -> bool> Predicate<T, F> {
    /// XXX TODO placeholder
    pub fn test(&mut self, element: T) -> bool { self.0(element) }

    /// XXX TODO placeholder
    pub fn as_fn(&mut self) -> &mut F { &mut self.0 }
}


/// Calls callback until it returns `SearchStep::Done` or either `Match` or
/// `Reject` depending no `MATCH` generic argument.
pub(super) fn loop_next<const MATCH: bool, T>(
    mut next: impl FnMut() -> SearchStep<T>,
) -> Option<(T, T)> {
    loop {
        match next() {
            SearchStep::Done => break None,
            SearchStep::Match(start, end) if MATCH => break Some((start, end)),
            SearchStep::Reject(start, end) if !MATCH => break Some((start, end)),
            _ => (),
        }
    }
}
