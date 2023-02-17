#![unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]

use core::pattern::{Haystack, Pattern, SearchStep};
use core::pattern;
use core::str::try_first_code_point;

#[derive(Debug)]
pub struct Slice {
    pub inner: [u8],
}

impl Slice {
    #[inline]
    fn from_u8_slice(s: &[u8]) -> &Slice {
        unsafe { core::mem::transmute(s) }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for Haystack
/////////////////////////////////////////////////////////////////////////////

impl<'hs> Haystack for &'hs Slice {
    type Cursor = usize;

    fn cursor_at_front(&self) -> usize { 0 }
    fn cursor_at_back(&self) -> usize { self.inner.len() }

    unsafe fn split_at_cursor_unchecked(self, pos: usize) -> (Self, Self) {
        // SAFETY: Caller promises cursor is valid.
        unsafe { (get_unchecked(&self, ..pos), get_unchecked(&self, pos..)) }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl Pattern for char
/////////////////////////////////////////////////////////////////////////////

impl<'hs> Pattern<&'hs Slice> for char {
    type Searcher = CharSearcher<'hs>;

    fn into_searcher(self, slice: &'hs Slice) -> Self::Searcher {
        Self::Searcher::new(slice, self)
    }

    fn is_contained_in(self, slice: &'hs Slice) -> bool {
        let mut buf = [0; 4];
        slice.inner.contains(self.encode_utf8(&mut buf).as_bytes())
    }

    fn is_prefix_of(self, slice: &'hs Slice) -> bool {
        let mut buf = [0; 4];
        slice.inner.starts_with(self.encode_utf8(&mut buf).as_bytes())
    }

    fn is_suffix_of(self, slice: &'hs Slice) -> bool {
        let mut buf = [0; 4];
        slice.inner.ends_with(self.encode_utf8(&mut buf).as_bytes())
    }

    fn strip_prefix_of(self, slice: &'hs Slice) -> Option<&'hs Slice> {
        let mut buf = [0; 4];
        let needle = self.encode_utf8(&mut buf).as_bytes();
        slice.inner.starts_with(needle).then(|| {
            // SAFETY: We’ve just checked slice starts with needle.
            unsafe { get_unchecked(slice, needle.len()..) }
        })
    }

    fn strip_suffix_of(self, slice: &'hs Slice) -> Option<&'hs Slice> {
        let mut buf = [0; 4];
        let needle = self.encode_utf8(&mut buf).as_bytes();
        slice.inner.ends_with(needle).then(|| {
            // SAFETY: We’ve just checked slice starts with needle.
            unsafe { get_unchecked(slice, ..slice.inner.len() - needle.len()) }
        })
    }
}

#[derive(Clone, Debug)]
pub struct CharSearcher<'hs> {
    /// Zero-padded UTF-8 encoded character we’re searching for.
    _needle: Box<[u8; 4]>,
    /// Slice searcher over the slice.
    searcher: <&'hs [u8] as Pattern<&'hs [u8]>>::Searcher,
}

impl<'hs> CharSearcher<'hs> {
    fn new(slice: &'hs Slice, chr: char) -> Self {
        let mut buf = [0; 4];
        let len = chr.encode_utf8(&mut buf).len();
        let needle = Box::new(buf);
        // XXX: This is potentially unsound?  We’re transmuting needle’s
        // lifetime to 'hs which is definitely not true, but at the same time
        // Searcher dies when needle dies so it won’t reference it after it
        // dies.
        let pattern: &'hs [u8] = unsafe { core::mem::transmute(&needle[..len]) };
        Self {
            _needle: needle,
            searcher: pattern.into_searcher(&slice.inner)
        }
    }
}

unsafe impl<'hs> pattern::Searcher<&'hs Slice> for CharSearcher<'hs> {
    fn haystack(&self) -> &'hs Slice {
        Slice::from_u8_slice(self.searcher.haystack())
    }

    fn next(&mut self) -> SearchStep<usize> {
        self.searcher.next()
    }

    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.searcher.next_match()
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.searcher.next_match()
    }
}

unsafe impl<'hs> pattern::ReverseSearcher<&'hs Slice> for CharSearcher<'hs> {
    fn next_back(&mut self) -> SearchStep<usize> {
        self.searcher.next_back()
    }

    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.searcher.next_match_back()
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.searcher.next_match_back()
    }
}

impl<'hs> pattern::DoubleEndedSearcher<&'hs Slice> for CharSearcher<'hs> {}

/////////////////////////////////////////////////////////////////////////////
// Impl Pattern for &FnMut(char)
/////////////////////////////////////////////////////////////////////////////

// XXX TODO
// This is work-around of the following:
//     error[E0210]: type parameter `F` must be covered by another type when it
//                   appears before the first local type (`pattern::Slice`)
//        --> library/std/src/sys/unix/os_str/pattern.rs:148:11
//         |
//     148 | impl<'hs, F: FnMut(char) -> bool> Pattern<&'hs Slice> for F {
//         |           ^ type parameter `F` must be covered by another type when
//                       it appears before the first local type (`pattern::Slice`)
//         |
pub struct Predicate<F>(F);

#[rustc_has_incoherent_inherent_impls]
impl<'hs, F: FnMut(char) -> bool> Pattern<&'hs Slice> for F {
    type Searcher = PredicateSearcher<'hs, F>;

    fn into_searcher(self, slice: &'hs Slice) -> Self::Searcher {
        Self::Searcher::new(slice, self)
    }

    fn is_prefix_of(mut self, slice: &'hs Slice) -> bool {
        matches!(try_first_code_point(&slice.inner),
                 Some((chr, _)) if self(chr))
    }

    fn is_suffix_of(mut self, slice: &'hs Slice) -> bool {
        matches!(try_last_code_point(&slice.inner),
                 Some((chr, _)) if self(chr))
    }

    fn strip_prefix_of(mut self, slice: &'hs Slice) -> Option<&'hs Slice> {
        let bytes = &slice.inner;
        if let Some((chr, len)) = try_first_code_point(bytes) {
            if self(chr) {
                return Some(Slice::from_u8_slice(&bytes[len..]));
            }
        }
        None
    }

    fn strip_suffix_of(mut self, slice: &'hs Slice) -> Option<&'hs Slice> {
        let bytes = &slice.inner;
        if let Some((chr, len)) = try_last_code_point(bytes) {
            if self(chr) {
                return Some(Slice::from_u8_slice(&bytes[..bytes.len() - len]));
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct PredicateSearcher<'hs, F> {
    slice: &'hs Slice,
    pred: F,

    start: usize,
    end: usize,
    fw_match_len: usize,
    bw_match_len: usize,
}

impl<'hs, F: FnMut(char) -> bool> PredicateSearcher<'hs, F> {
    fn new(slice: &'hs Slice, pred: F) -> Self {
        Self {
            slice: slice,
            pred,
            start: 0,
            end: 0,
            fw_match_len: 0,
            bw_match_len: 0,
        }
    }

    /// Looks for the next match and returns its position and length.  Doesn’t
    /// update searcher’s state.
    fn next_match_impl(&mut self) -> Option<(usize, usize)> {
        let bytes = &self.slice.inner[..self.end];
        let mut pos = self.start;
        while pos < bytes.len() {
            pos += count_utf8_cont_bytes(bytes[pos..].iter());
            if let Some((chr, len)) = try_first_code_point(&bytes[pos..]) {
                if (self.pred)(chr) {
                    return Some((pos, len))
                }
                pos += len;
            } else {
                pos += 1;
            }
        }
        None
    }

    /// Implementation of Searcher::next and Searcher::next_match functions.
    fn next_impl<R: SearchReturn>(&mut self) -> R {
        while self.start < self.end {
            if self.fw_match_len == 0 {
                let (pos, len) = self.next_match_impl().unwrap_or((self.end, 0));
                self.fw_match_len = len;
                let start = self.start;
                if pos != start {
                    self.start = pos;
                    if let Some(ret) = R::rejecting(start, pos) {
                        return ret;
                    }
                }
            }

            debug_assert_ne!(0, self.fw_match_len);
            let pos = self.start;
            self.start += self.fw_match_len;
            self.fw_match_len = 0;
            if let Some(ret) = R::matching(pos, self.start) {
                return ret;
            }
        }
        R::DONE
    }

    /// Looks for the next match back and returns its position and length.
    /// Doesn’t update searcher’s state.
    fn next_match_back_impl(&mut self) -> Option<(usize, usize)> {
        let mut bytes = &self.slice.inner[self.start..self.end];
        while !bytes.is_empty() {
            let pos = bytes.len() - count_utf8_cont_bytes(bytes.iter().rev());
            let pos = pos.checked_sub(1)?;
            if let Some((chr, len)) = try_first_code_point(&bytes[pos..]) {
                if (self.pred)(chr) {
                    return Some((pos + self.start, len))
                }
            }
            bytes = &bytes[..pos]
        }
        None
    }

    /// Implementation of ReverseSearcher::next and ReverseSearcher::next_match
    /// functions.
    fn next_back_impl<R: SearchReturn>(&mut self) -> R {
        while self.start < self.end {
            if self.bw_match_len == 0 {
                let end = self.end;
                let (pos, len) = self.next_match_back_impl().unwrap_or((end, 0));
                self.bw_match_len = len;
                if pos + len != end {
                    self.end = pos + len;
                    if let Some(ret) = R::rejecting(self.end, end) {
                        return ret;
                    }
                }
            }

            debug_assert_ne!(0, self.bw_match_len);
            let end = self.end;
            self.end -= self.bw_match_len;
            self.bw_match_len = 0;
            if let Some(ret) = R::matching(self.end, end) {
                return ret;
            }
        }
        R::DONE
    }
}

unsafe impl<'hs, F: FnMut(char) -> bool> pattern::Searcher<&'hs Slice> for PredicateSearcher<'hs, F> {
    fn haystack(&self) -> &'hs Slice { self.slice }

    fn next(&mut self) -> SearchStep<usize> {
        self.next_impl()
    }

    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.next_impl::<MatchOnly>().0
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.next_impl::<RejectOnly>().0
    }
}

unsafe impl<'hs, F: FnMut(char) -> bool> pattern::ReverseSearcher<&'hs Slice> for PredicateSearcher<'hs, F> {
    fn next_back(&mut self) -> SearchStep<usize> {
        self.next_back_impl()
    }

    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.next_back_impl::<MatchOnly>().0
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.next_back_impl::<RejectOnly>().0
    }
}

impl<'hs, F: FnMut(char) -> bool> pattern::DoubleEndedSearcher<&'hs Slice> for PredicateSearcher<'hs, F> {}

/////////////////////////////////////////////////////////////////////////////

/// Possible return type of a search.
///
/// It abstract differences between `next`, `next_match` and `next_reject`
/// methods.  Depending on return type an implementation for those functions
/// will generate matches and rejects, only matches or only rejects.
trait SearchReturn: Sized {
    const DONE: Self;
    fn matching(start: usize, end: usize) -> Option<Self>;
    fn rejecting(start: usize, end: usize) -> Option<Self>;
}

struct MatchOnly(Option<(usize, usize)>);
struct RejectOnly(Option<(usize, usize)>);

impl SearchReturn for SearchStep<usize> {
    const DONE: Self = SearchStep::Done;
    fn matching(s: usize, e: usize) -> Option<Self> {
        Some(SearchStep::Match(s, e))
    }
    fn rejecting(s: usize, e: usize) ->Option<Self> {
        Some(SearchStep::Reject(s, e))
    }
}

impl SearchReturn for MatchOnly {
    const DONE: Self = Self(None);
    fn matching(s: usize, e: usize) -> Option<Self> { Some(Self(Some((s, e)))) }
    fn rejecting(_s: usize, _e: usize) -> Option<Self> { None }
}

impl SearchReturn for RejectOnly {
    const DONE: Self = Self(None);
    fn matching(_s: usize, _e: usize) -> Option<Self> { None }
    fn rejecting(s: usize, e: usize) -> Option<Self> { Some(Self(Some((s, e)))) }
}


unsafe fn get_unchecked<I>(slice: &Slice, index: I) -> &Slice
where I: core::slice::SliceIndex<[u8], Output = [u8]>,
{
    // SAFETY: Caller Promises index is valid.
    Slice::from_u8_slice(unsafe { slice.inner.get_unchecked(index) })
}


/// Tries to extract UTF-8 sequence from the end of the slice.
///
/// If last bytes of the slice don’t form a valid UTF-8 sequence (or if slice is
/// empty), returns `None`.  If they do, decodes the character and returns its
/// encoded length.
fn try_last_code_point(bytes: &[u8]) -> Option<(char, usize)> {
    // Fast path: ASCII
    let last = *bytes.last()?;
    if last < 0x80 {
        return Some((unsafe { char::from_u32_unchecked(last as u32) }, 1));
    }

    // Count how many continuation bytes there are at the end.
    let count = count_utf8_cont_bytes(bytes.iter().rev().take(4));
    if count == bytes.len() || count >= 4 {
        return None;
    }
    let pos = bytes.len() - count - 1;

    // Try decode.  If length matches, we have ourselves a character.
    let (chr, len) = try_first_code_point(&bytes[pos..])?;
    (len == count + 1).then_some((chr, len))
}


/// Counts UTF-8 continuation bytes at the beginning of the iterator.
#[inline]
fn count_utf8_cont_bytes<'a>(bytes: impl Iterator<Item = &'a u8>) -> usize {
    bytes.take_while(|&&byte| (byte as i8) < -64).count()
}
