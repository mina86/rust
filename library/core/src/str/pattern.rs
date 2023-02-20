//! [The Pattern API] implementation for searching in `&str`.
//!
//! The implementation provides generic mechanism for using different pattern
//! types when searching through a string.  Although this API is unstable, it is
//! exposed via stable APIs on the [`str`] type.
//!
//! Depending on the type of the pattern, the behaviour of methods like
//! [`str::find`] and [`str::contains`] can change. The table below describes
//! some of those behaviours.
//!
//! | Pattern type             | Match condition                           |
//! |--------------------------|-------------------------------------------|
//! | `&str`                   | is substring                              |
//! | `char`                   | is contained in string                    |
//! | `&[char]`                | any char in slice is contained in string  |
//! | `F: FnMut(char) -> bool` | `F` returns `true` for a char in string   |
//! | `&&str`                  | is substring                              |
//! | `&String`                | is substring                              |
//!
//! # Examples
//!
//! ```
//! let s = "Can you find a needle in a haystack?";
//!
//! // &str pattern
//! assert_eq!(s.find("you"), Some(4));
//! assert_eq!(s.find("thou"), None);
//!
//! // char pattern
//! assert_eq!(s.find('n'), Some(2));
//! assert_eq!(s.find('N'), None);
//!
//! // Array of chars pattern and slices thereof
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u']), Some(1));
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u'][..]), Some(1));
//! assert_eq!(s.find(&['q', 'v', 'x']), None);
//!
//! // Predicate closure
//! assert_eq!(s.find(|c: char| c.is_ascii_punctuation()), Some(35));
//! assert_eq!(s.find(|c: char| c.is_lowercase()), Some(1));
//! assert_eq!(s.find(|c: char| !c.is_ascii()), None);
//! ```
//!
//! [The Pattern API]: crate::pattern

#![unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]

use crate::fmt;
use crate::ops::Range;
use crate::pattern::{
    DoubleEndedSearcher, Haystack, Pattern, ReverseSearcher, SearchStep, Searcher,
};
use crate::str_bytes;

/////////////////////////////////////////////////////////////////////////////
// Impl for Haystack
/////////////////////////////////////////////////////////////////////////////

impl<'a> Haystack for &'a str {
    type Cursor = usize;

    #[inline(always)]
    fn cursor_at_front(self) -> usize {
        0
    }
    #[inline(always)]
    fn cursor_at_back(self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.is_empty()
    }

    #[inline(always)]
    unsafe fn get_unchecked(self, range: Range<usize>) -> Self {
        // SAFETY: Caller promises position is a character boundary.
        unsafe { self.get_unchecked(range) }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for char
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<char as Pattern<H>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharSearcher<'a>(str_bytes::CharSearcher<'a, str_bytes::Utf8>);

impl<'a> CharSearcher<'a> {
    fn new(haystack: &'a str, chr: char) -> Self {
        Self(str_bytes::CharSearcher::new(str_bytes::Bytes::from(haystack), chr))
    }
}

unsafe impl<'a> Searcher<&'a str> for CharSearcher<'a> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.0.haystack().into()
    }
    #[inline]
    fn next(&mut self) -> SearchStep {
        self.0.next()
    }
    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.0.next_match()
    }
    #[inline]
    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject()
    }
}

unsafe impl<'a> ReverseSearcher<&'a str> for CharSearcher<'a> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        self.0.next_back()
    }
    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_match_back()
    }
    #[inline]
    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject_back()
    }
}

impl<'a> DoubleEndedSearcher<&'a str> for CharSearcher<'a> {}

/// Searches for chars that are equal to a given [`char`].
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find('o'), Some(4));
/// assert_eq!("Hello world".find('x'), None);
/// ```
impl<'a> Pattern<&'a str> for char {
    type Searcher = CharSearcher<'a>;

    #[inline]
    fn into_searcher(self, haystack: &'a str) -> Self::Searcher {
        CharSearcher::new(haystack, self)
    }

    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        self.is_contained_in(str_bytes::Bytes::from(haystack))
    }

    #[inline]
    fn is_prefix_of(self, haystack: &'a str) -> bool {
        self.is_prefix_of(str_bytes::Bytes::from(haystack))
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &'a str) -> Option<&'a str> {
        self.strip_prefix_of(str_bytes::Bytes::from(haystack)).map(<&str>::from)
    }

    #[inline]
    fn is_suffix_of(self, haystack: &'a str) -> bool {
        self.is_suffix_of(str_bytes::Bytes::from(haystack))
    }

    #[inline]
    fn strip_suffix_of(self, haystack: &'a str) -> Option<&'a str> {
        self.strip_suffix_of(str_bytes::Bytes::from(haystack)).map(<&str>::from)
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for a MultiCharEq wrapper
/////////////////////////////////////////////////////////////////////////////

#[doc(hidden)]
trait MultiCharEq {
    fn matches(&mut self, c: char) -> bool;
}

impl<F> MultiCharEq for F
where
    F: FnMut(char) -> bool,
{
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        (*self)(c)
    }
}

impl<const N: usize> MultiCharEq for [char; N] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&m| m == c)
    }
}

impl<const N: usize> MultiCharEq for &[char; N] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&m| m == c)
    }
}

impl MultiCharEq for &[char] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&m| m == c)
    }
}

struct MultiCharEqPattern<C: MultiCharEq>(C);

#[derive(Clone, Debug)]
struct MultiCharEqSearcher<'a, C: MultiCharEq> {
    char_eq: C,
    haystack: &'a str,
    char_indices: super::CharIndices<'a>,
}

impl<'a, C: MultiCharEq> Pattern<&'a str> for MultiCharEqPattern<C> {
    type Searcher = MultiCharEqSearcher<'a, C>;

    #[inline]
    fn into_searcher(self, haystack: &'a str) -> MultiCharEqSearcher<'a, C> {
        MultiCharEqSearcher { haystack, char_eq: self.0, char_indices: haystack.char_indices() }
    }
}

unsafe impl<'a, C: MultiCharEq> Searcher<&'a str> for MultiCharEqSearcher<'a, C> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        let s = &mut self.char_indices;
        // Compare lengths of the internal byte slice iterator
        // to find length of current char
        let pre_len = s.iter.iter.len();
        if let Some((i, c)) = s.next() {
            let len = s.iter.iter.len();
            let char_len = pre_len - len;
            if self.char_eq.matches(c) {
                return SearchStep::Match(i, i + char_len);
            } else {
                return SearchStep::Reject(i, i + char_len);
            }
        }
        SearchStep::Done
    }
}

unsafe impl<'a, C: MultiCharEq> ReverseSearcher<&'a str> for MultiCharEqSearcher<'a, C> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        let s = &mut self.char_indices;
        // Compare lengths of the internal byte slice iterator
        // to find length of current char
        let pre_len = s.iter.iter.len();
        if let Some((i, c)) = s.next_back() {
            let len = s.iter.iter.len();
            let char_len = pre_len - len;
            if self.char_eq.matches(c) {
                return SearchStep::Match(i, i + char_len);
            } else {
                return SearchStep::Reject(i, i + char_len);
            }
        }
        SearchStep::Done
    }
}

impl<'a, C: MultiCharEq> DoubleEndedSearcher<&'a str> for MultiCharEqSearcher<'a, C> {}

/////////////////////////////////////////////////////////////////////////////

macro_rules! pattern_methods {
    ($t:ty, $pmap:expr, $smap:expr) => {
        type Searcher = $t;

        #[inline]
        fn into_searcher(self, haystack: &'a str) -> $t {
            ($smap)(($pmap)(self).into_searcher(haystack))
        }

        #[inline]
        fn is_contained_in(self, haystack: &'a str) -> bool {
            ($pmap)(self).is_contained_in(haystack)
        }

        #[inline]
        fn is_prefix_of(self, haystack: &'a str) -> bool {
            ($pmap)(self).is_prefix_of(haystack)
        }

        #[inline]
        fn strip_prefix_of(self, haystack: &'a str) -> Option<&'a str> {
            ($pmap)(self).strip_prefix_of(haystack)
        }

        #[inline]
        fn is_suffix_of(self, haystack: &'a str) -> bool
        where
            $t: ReverseSearcher<&'a str>,
        {
            ($pmap)(self).is_suffix_of(haystack)
        }

        #[inline]
        fn strip_suffix_of(self, haystack: &'a str) -> Option<&'a str>
        where
            $t: ReverseSearcher<&'a str>,
        {
            ($pmap)(self).strip_suffix_of(haystack)
        }
    };
}

macro_rules! searcher_methods {
    (forward) => {
        #[inline]
        fn haystack(&self) -> &'a str {
            self.0.haystack()
        }
        #[inline]
        fn next(&mut self) -> SearchStep {
            self.0.next()
        }
        #[inline]
        fn next_match(&mut self) -> Option<(usize, usize)> {
            self.0.next_match()
        }
        #[inline]
        fn next_reject(&mut self) -> Option<(usize, usize)> {
            self.0.next_reject()
        }
    };
    (reverse) => {
        #[inline]
        fn next_back(&mut self) -> SearchStep {
            self.0.next_back()
        }
        #[inline]
        fn next_match_back(&mut self) -> Option<(usize, usize)> {
            self.0.next_match_back()
        }
        #[inline]
        fn next_reject_back(&mut self) -> Option<(usize, usize)> {
            self.0.next_reject_back()
        }
    };
}

/// Associated type for `<[char; N] as Pattern<&'a str>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharArraySearcher<'a, const N: usize>(
    <MultiCharEqPattern<[char; N]> as Pattern<&'a str>>::Searcher,
);

/// Associated type for `<&[char; N] as Pattern<&'a str>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharArrayRefSearcher<'a, 'b, const N: usize>(
    <MultiCharEqPattern<&'b [char; N]> as Pattern<&'a str>>::Searcher,
);

/// Searches for chars that are equal to any of the [`char`]s in the array.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(['l', 'l']), Some(2));
/// assert_eq!("Hello world".find(['l', 'l']), Some(2));
/// ```
impl<'a, const N: usize> Pattern<&'a str> for [char; N] {
    pattern_methods!(CharArraySearcher<'a, N>, MultiCharEqPattern, CharArraySearcher);
}

unsafe impl<'a, const N: usize> Searcher<&'a str> for CharArraySearcher<'a, N> {
    searcher_methods!(forward);
}

unsafe impl<'a, const N: usize> ReverseSearcher<&'a str> for CharArraySearcher<'a, N> {
    searcher_methods!(reverse);
}

/// Searches for chars that are equal to any of the [`char`]s in the array.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(&['l', 'l']), Some(2));
/// assert_eq!("Hello world".find(&['l', 'l']), Some(2));
/// ```
impl<'a, 'b, const N: usize> Pattern<&'a str> for &'b [char; N] {
    pattern_methods!(CharArrayRefSearcher<'a, 'b, N>, MultiCharEqPattern, CharArrayRefSearcher);
}

unsafe impl<'a, 'b, const N: usize> Searcher<&'a str> for CharArrayRefSearcher<'a, 'b, N> {
    searcher_methods!(forward);
}

unsafe impl<'a, 'b, const N: usize> ReverseSearcher<&'a str> for CharArrayRefSearcher<'a, 'b, N> {
    searcher_methods!(reverse);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &[char]
/////////////////////////////////////////////////////////////////////////////

// Todo: Change / Remove due to ambiguity in meaning.

/// Associated type for `<&[char] as Pattern<&'a str>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharSliceSearcher<'a, 'b>(
    <MultiCharEqPattern<&'b [char]> as Pattern<&'a str>>::Searcher,
);

unsafe impl<'a, 'b> Searcher<&'a str> for CharSliceSearcher<'a, 'b> {
    searcher_methods!(forward);
}

unsafe impl<'a, 'b> ReverseSearcher<&'a str> for CharSliceSearcher<'a, 'b> {
    searcher_methods!(reverse);
}

impl<'a, 'b> DoubleEndedSearcher<&'a str> for CharSliceSearcher<'a, 'b> {}

/// Searches for chars that are equal to any of the [`char`]s in the slice.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(&['l', 'l'] as &[_]), Some(2));
/// assert_eq!("Hello world".find(&['l', 'l'][..]), Some(2));
/// ```
impl<'a, 'b> Pattern<&'a str> for &'b [char] {
    pattern_methods!(CharSliceSearcher<'a, 'b>, MultiCharEqPattern, CharSliceSearcher);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for F: FnMut(char) -> bool
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<F as Pattern<&'a str>>::Searcher`.
#[derive(Clone)]
pub struct CharPredicateSearcher<'a, F>(<MultiCharEqPattern<F> as Pattern<&'a str>>::Searcher)
where
    F: FnMut(char) -> bool;

impl<F> fmt::Debug for CharPredicateSearcher<'_, F>
where
    F: FnMut(char) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharPredicateSearcher")
            .field("haystack", &self.0.haystack)
            .field("char_indices", &self.0.char_indices)
            .finish()
    }
}
unsafe impl<'a, F> Searcher<&'a str> for CharPredicateSearcher<'a, F>
where
    F: FnMut(char) -> bool,
{
    searcher_methods!(forward);
}

unsafe impl<'a, F> ReverseSearcher<&'a str> for CharPredicateSearcher<'a, F>
where
    F: FnMut(char) -> bool,
{
    searcher_methods!(reverse);
}

impl<'a, F> DoubleEndedSearcher<&'a str> for CharPredicateSearcher<'a, F> where
    F: FnMut(char) -> bool
{
}

/// Searches for [`char`]s that match the given predicate.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(char::is_uppercase), Some(0));
/// assert_eq!("Hello world".find(|c| "aeiou".contains(c)), Some(1));
/// ```
impl<'a, F> Pattern<&'a str> for F
where
    F: FnMut(char) -> bool,
{
    pattern_methods!(CharPredicateSearcher<'a, F>, MultiCharEqPattern, CharPredicateSearcher);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &&str
/////////////////////////////////////////////////////////////////////////////

/// Delegates to the `&str` impl.
impl<'a, 'b, 'c> Pattern<&'a str> for &'c &'b str {
    pattern_methods!(StrSearcher<'a, 'b>, |&s| s, |s| s);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &str
/////////////////////////////////////////////////////////////////////////////

/// Non-allocating substring search.
///
/// Will handle the pattern `""` as returning empty matches at each character
/// boundary.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find("world"), Some(6));
/// ```
impl<'a, 'b> Pattern<&'a str> for &'b str {
    type Searcher = StrSearcher<'a, 'b>;

    #[inline]
    fn into_searcher(self, haystack: &'a str) -> StrSearcher<'a, 'b> {
        StrSearcher::new(haystack, self)
    }

    /// Checks whether the pattern matches at the front of the haystack.
    #[inline]
    fn is_prefix_of(self, haystack: &'a str) -> bool {
        haystack.as_bytes().starts_with(self.as_bytes())
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        self.as_bytes().is_contained_in(haystack.as_bytes())
    }

    /// Removes the pattern from the front of haystack, if it matches.
    #[inline]
    fn strip_prefix_of(self, haystack: &'a str) -> Option<&'a str> {
        if self.is_prefix_of(haystack) {
            // SAFETY: prefix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(self.as_bytes().len()..)) }
        } else {
            None
        }
    }

    /// Checks whether the pattern matches at the back of the haystack.
    #[inline]
    fn is_suffix_of(self, haystack: &'a str) -> bool {
        haystack.as_bytes().ends_with(self.as_bytes())
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of(self, haystack: &'a str) -> Option<&'a str> {
        if self.is_suffix_of(haystack) {
            let i = haystack.len() - self.as_bytes().len();
            // SAFETY: suffix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(..i)) }
        } else {
            None
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Two Way substring searcher
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
/// Associated type for `<&str as Pattern<&'a str>>::Searcher`.
pub struct StrSearcher<'a, 'b>(crate::str_bytes::StrSearcher<'a, 'b, crate::str_bytes::Utf8>);

impl<'a, 'b> StrSearcher<'a, 'b> {
    fn new(haystack: &'a str, needle: &'b str) -> StrSearcher<'a, 'b> {
        let haystack = crate::str_bytes::Bytes::from(haystack);
        Self(crate::str_bytes::StrSearcher::new(haystack, needle))
    }
}

unsafe impl<'a, 'b> Searcher<&'a str> for StrSearcher<'a, 'b> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.0.haystack().into()
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        self.0.next()
    }

    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.0.next_match()
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject()
    }
}

unsafe impl<'a, 'b> ReverseSearcher<&'a str> for StrSearcher<'a, 'b> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        self.0.next_back()
    }

    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_match_back()
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject_back()
    }
}
