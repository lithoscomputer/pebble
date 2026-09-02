//! Turning a fetched page into text a model can read.
//!
//! This is a deliberately small converter: it keeps the structure a reader
//! needs — headings, paragraphs, list items, line breaks, links and
//! preformatted blocks — and lets every other element contribute its text.
//! It is not an HTML parser and does not try to be one, because pebble takes
//! no dependency it cannot share with the model client it is built on, and a
//! conforming HTML5 parser is a large tree of them.
//!
//! Content that does not look like markup is passed through untouched, so a
//! JSON API response or a plain-text file reaches the model exactly as it was
//! served.

use std::borrow::Cow;
use std::mem::take;

/// Whether `text` looks like it holds HTML markup.
fn looks_like_html(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("<!")
        || trimmed.starts_with("<html")
        || trimmed.starts_with("<HTML")
        || trimmed.contains("</div>")
        || trimmed.contains("</p>")
        || trimmed.contains("</body>")
}

/// Converts HTML to Markdown, dropping scripts and styles.
///
/// Anything that does not look like HTML is returned unchanged.
pub(crate) fn html_to_markdown(text: &str) -> String {
    if !looks_like_html(text) {
        return text.to_owned();
    }
    Converter::default().convert(text)
}

/// A link being collected: where its text starts, and where it points.
struct Anchor {
    text_start: usize,
    href:       Option<String>,
}

/// What one finished block is, which decides how much room follows it.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum BlockKind {
    /// A paragraph, a heading, anything standing on its own.
    #[default]
    Standalone,
    /// One item of a list, which sits directly under the item before it.
    Item,
}

/// The Markdown being built, one block at a time.
#[derive(Default)]
struct Converter {
    /// Finished blocks, joined at the end.
    blocks:       Vec<(BlockKind, String)>,
    /// The block being collected.
    current:      String,
    /// What the block being collected starts with: a heading's hashes, a list
    /// item's dash, or nothing.
    prefix:       String,
    /// What the block being collected is.
    kind:         BlockKind,
    /// Whether collapsed whitespace is owed before the next text.
    owes_space:   bool,
    /// The link being collected, if any.
    anchor:       Option<Anchor>,
    /// How many `<pre>` elements are open; inside one, whitespace is kept.
    preformatted: usize,
}

impl Converter {
    fn convert(mut self, text: &str) -> String {
        let mut index = 0;
        while let Some(offset) = text[index..].find('<') {
            let tag_start = index + offset;
            self.push_text(&text[index..tag_start]);
            index = self.consume_markup(text, tag_start);
        }
        self.push_text(&text[index..]);
        self.flush();
        self.render()
    }

    /// Joins the finished blocks, keeping the items of a list together.
    fn render(&self) -> String {
        let mut rendered = String::new();
        let mut previous: Option<BlockKind> = None;
        for (kind, block) in &self.blocks {
            match previous {
                None => {}
                Some(BlockKind::Item) if *kind == BlockKind::Item => rendered.push('\n'),
                Some(_) => rendered.push_str("\n\n"),
            }
            rendered.push_str(block);
            previous = Some(*kind);
        }
        rendered
    }

    /// Handles the markup starting at `start`, and answers where to resume.
    fn consume_markup(&mut self, text: &str, start: usize) -> usize {
        let rest = &text[start..];
        if rest.starts_with("<!--") {
            return skip_past(text, start + 4, "-->");
        }
        if rest.starts_with("<!") {
            return skip_past(text, start + 2, ">");
        }
        let Some(tag) = Tag::parse(text, start) else {
            // A bare `<` that opens nothing is content.
            self.push_text("<");
            return start + 1;
        };
        if !tag.closing && matches!(tag.name.as_str(), "script" | "style") {
            return skip_raw_text(text, tag.end, &tag.name);
        }
        self.handle(&tag);
        tag.end
    }

    /// Applies one tag to the block being built.
    fn handle(&mut self, tag: &Tag) {
        match tag.name.as_str() {
            "br" => {
                self.owes_space = false;
                self.current.push('\n');
            }
            "hr" => {
                self.flush();
                self.blocks.push((BlockKind::Standalone, "---".to_owned()));
            }
            "a" => {
                if tag.closing {
                    self.close_anchor();
                } else {
                    self.open_anchor(tag.attribute("href"));
                }
            }
            "pre" => {
                self.flush();
                if tag.closing {
                    self.preformatted = self.preformatted.saturating_sub(1);
                } else {
                    self.preformatted += 1;
                }
            }
            "li" => {
                self.flush();
                if !tag.closing {
                    "- ".clone_into(&mut self.prefix);
                    self.kind = BlockKind::Item;
                }
            }
            "td" | "th" => self.owes_space = true,
            name => {
                if let Some(level) = heading_level(name) {
                    self.flush();
                    if !tag.closing {
                        self.prefix = format!("{} ", "#".repeat(level));
                    }
                } else if is_block(name) {
                    self.flush();
                }
            }
        }
    }

    /// Adds page text to the block being built.
    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let decoded = decode_entities(text);
        if self.preformatted > 0 {
            self.current.push_str(&decoded);
            return;
        }
        for character in decoded.chars() {
            if character.is_whitespace() {
                self.owes_space = !self.current.is_empty();
            } else {
                if self.owes_space && !self.current.is_empty() {
                    self.current.push(' ');
                }
                self.owes_space = false;
                self.current.push(character);
            }
        }
    }

    fn open_anchor(&mut self, href: Option<String>) {
        if self.owes_space && !self.current.is_empty() {
            self.current.push(' ');
            self.owes_space = false;
        }
        self.anchor = Some(Anchor {
            text_start: self.current.len(),
            href,
        });
    }

    /// Rewrites the collected anchor text as a Markdown link.
    ///
    /// A link with no destination, or with no text, stays as the text it has:
    /// `[](https://example.com)` reads as nothing at all.
    fn close_anchor(&mut self) {
        let Some(anchor) = self.anchor.take() else {
            return;
        };
        let Some(href) = anchor.href.filter(|href| !href.is_empty()) else {
            return;
        };
        if anchor.text_start > self.current.len() {
            return;
        }
        let text = self.current.split_off(anchor.text_start);
        if text.is_empty() {
            return;
        }
        self.current.push('[');
        self.current.push_str(&text);
        self.current.push_str("](");
        self.current.push_str(&href);
        self.current.push(')');
    }

    /// Ends the block being built, keeping it when it has any text.
    fn flush(&mut self) {
        self.close_anchor();
        self.owes_space = false;
        let text = self.current.trim();
        if !text.is_empty() {
            let mut block = take(&mut self.prefix);
            block.push_str(text);
            self.blocks.push((self.kind, block));
        }
        self.current.clear();
        self.prefix.clear();
        self.kind = BlockKind::Standalone;
    }
}

/// The heading level `name` names, if it is a heading.
fn heading_level(name: &str) -> Option<usize> {
    let level = name.strip_prefix('h')?;
    match level {
        "1" | "2" | "3" | "4" | "5" | "6" => level.parse().ok(),
        _ => None,
    }
}

/// Whether an element ends the block around it.
fn is_block(name: &str) -> bool {
    matches!(
        name,
        "address"
            | "article"
            | "aside"
            | "blockquote"
            | "body"
            | "dd"
            | "details"
            | "div"
            | "dl"
            | "dt"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "head"
            | "header"
            | "html"
            | "main"
            | "nav"
            | "ol"
            | "p"
            | "section"
            | "summary"
            | "table"
            | "tbody"
            | "tfoot"
            | "thead"
            | "title"
            | "tr"
            | "ul"
    )
}

/// Where `text` continues after the next `terminator` at or after `from`.
fn skip_past(text: &str, from: usize, terminator: &str) -> usize {
    text[from..]
        .find(terminator)
        .map_or(text.len(), |offset| from + offset + terminator.len())
}

/// Where `text` continues after the closing tag of the `<script>` or
/// `<style>` element named `name` that is open at `from`.
///
/// The body of one of these elements is not markup, so the only thing that
/// ends it is its own closing tag, in whatever case the page wrote it and with
/// any whitespace before its `>`. A closing tag of some other element inside
/// the body is body text. A body that never closes takes the rest of the page
/// with it, which is what a browser does.
fn skip_raw_text(text: &str, from: usize, name: &str) -> usize {
    let mut index = from;
    while let Some(offset) = text[index..].find("</") {
        let start = index + offset;
        if let Some(tag) = Tag::parse(text, start)
            && tag.name == name
        {
            return tag.end;
        }
        index = start + 2;
    }
    text.len()
}

/// One parsed tag.
struct Tag {
    /// The element name, lowercased.
    name:       String,
    /// Whether this is a closing tag.
    closing:    bool,
    /// The text between the name and the `>`.
    attributes: String,
    /// Where the tag ends in the source.
    end:        usize,
}

impl Tag {
    /// Parses the tag starting at `start`, or `None` when what is there is not
    /// a tag.
    ///
    /// A tag that is never closed takes the rest of the page with it, which is
    /// what a browser does with `<div class="` at the end of a truncated
    /// response: the text after it was inside an attribute, not on the page.
    fn parse(text: &str, start: usize) -> Option<Self> {
        let after_bracket = &text[start + 1..];
        let closing = after_bracket.starts_with('/');
        let name_start = start + 1 + usize::from(closing);
        let name: String = text[name_start..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect();
        if name.is_empty() {
            return None;
        }

        let attributes_start = name_start + name.len();
        let mut close = None;
        let mut quote: Option<char> = None;
        for (offset, character) in text[attributes_start..].char_indices() {
            match (quote, character) {
                (Some(open), character) if character == open => quote = None,
                (None, '"' | '\'') => quote = Some(character),
                (None, '>') => {
                    close = Some(attributes_start + offset);
                    break;
                }
                _ => {}
            }
        }

        let (attributes_end, end) =
            close.map_or((text.len(), text.len()), |close| (close, close + 1));
        Some(Self {
            name: name.to_ascii_lowercase(),
            closing,
            attributes: text[attributes_start..attributes_end].to_owned(),
            end,
        })
    }

    /// The value of one attribute, when the tag carries it.
    ///
    /// Only quoted values are read, which is what a real page writes and what
    /// a link needs. The name has to stand on its own, so a link's `href` is
    /// never read out of the `data-href` beside it, and the same quote walk
    /// [`parse`](Self::parse) uses skips whatever another attribute's value
    /// says: a page that carries markup inside an attribute has an `href` in
    /// that text, not on this tag.
    fn attribute(&self, name: &str) -> Option<String> {
        let lowered = self.attributes.to_ascii_lowercase();
        let mut quote: Option<char> = None;
        for (start, character) in lowered.char_indices() {
            match (quote, character) {
                (Some(open), character) if character == open => quote = None,
                (Some(_), _) => {}
                (None, '"' | '\'') => quote = Some(character),
                (None, _) => {
                    let standalone = lowered[..start]
                        .chars()
                        .next_back()
                        .is_none_or(char::is_whitespace);
                    if standalone
                        && lowered[start..].starts_with(name)
                        && let Some(value) = self.quoted_value(start + name.len())
                    {
                        return Some(value);
                    }
                }
            }
        }
        None
    }

    /// The quoted value of the attribute whose name ends at `from`, when it has
    /// one.
    fn quoted_value(&self, from: usize) -> Option<String> {
        let value = self.attributes[from..].trim_start().strip_prefix('=')?;
        let value = value.trim_start();
        let quote = value.chars().next().filter(|c| *c == '"' || *c == '\'')?;
        let value = &value[quote.len_utf8()..];
        let end = value.find(quote)?;
        Some(decode_entities(&value[..end]).into_owned())
    }
}

/// Replaces the character references a page actually uses.
fn decode_entities(text: &str) -> Cow<'_, str> {
    if !text.contains('&') {
        return Cow::Borrowed(text);
    }

    let mut decoded = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(offset) = rest.find('&') {
        decoded.push_str(&rest[..offset]);
        let after = &rest[offset + 1..];
        // A reference is short and ends at a semicolon. Anything else — a bare
        // ampersand in prose, a query string — is the character itself.
        let resolved = after
            .find(';')
            .filter(|end| *end <= MAX_ENTITY_NAME_BYTES)
            .and_then(|end| entity(&after[..end]).map(|character| (character, end)));
        if let Some((character, end)) = resolved {
            decoded.push(character);
            rest = &after[end + 1..];
        } else {
            decoded.push('&');
            rest = after;
        }
    }
    decoded.push_str(rest);
    Cow::Owned(decoded)
}

/// How long a character reference's name may be before the ampersand is read
/// as prose.
const MAX_ENTITY_NAME_BYTES: usize = 10;

/// The character one reference names, if it is one this converter knows.
fn entity(name: &str) -> Option<char> {
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" | "#160" => Some(' '),
        "mdash" => Some('—'),
        "ndash" => Some('–'),
        "hellip" => Some('…'),
        _ => {
            let digits = name.strip_prefix('#')?;
            let code = match digits.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => digits.parse().ok()?,
            };
            char::from_u32(code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_heading_and_a_paragraph_become_markdown() {
        assert_eq!(
            html_to_markdown("<h1>Hello</h1><p>World</p>"),
            "# Hello\n\nWorld"
        );
    }

    #[test]
    fn every_heading_level_keeps_its_depth() {
        assert_eq!(
            html_to_markdown("<h2>Two</h2><h6>Six</h6><p>end</p>"),
            "## Two\n\n###### Six\n\nend"
        );
    }

    #[test]
    fn scripts_and_styles_contribute_nothing() {
        let html = "<html><head><style>body{color:red}</style></head><body><script>alert(1)\
                    </script><p>Content</p></body></html>";

        let markdown = html_to_markdown(html);

        assert!(!markdown.contains("alert"), "{markdown}");
        assert!(!markdown.contains("color:red"), "{markdown}");
        assert_eq!(markdown, "Content");
    }

    #[test]
    fn content_that_is_not_markup_passes_through() {
        let json = r#"{"key": "value", "items": [1, 2, 3]}"#;
        assert_eq!(html_to_markdown(json), json);

        let plain = "Just some plain text\nwith newlines";
        assert_eq!(html_to_markdown(plain), plain);
    }

    #[test]
    fn list_items_become_bullets_that_stay_together() {
        assert_eq!(
            html_to_markdown("<html><ul><li>one</li><li>two</li></ul><p>after</p></html>"),
            "- one\n- two\n\nafter"
        );
    }

    #[test]
    fn a_link_keeps_where_it_points() {
        assert_eq!(
            html_to_markdown(r#"<p>See <a href="https://example.com">the docs</a> first</p>"#),
            "See [the docs](https://example.com) first"
        );
    }

    #[test]
    fn a_link_without_a_destination_keeps_only_its_text() {
        assert_eq!(
            html_to_markdown("<p>See <a name=\"anchor\">the docs</a></p>"),
            "See the docs"
        );
    }

    #[test]
    fn inline_elements_stay_in_one_block() {
        assert_eq!(
            html_to_markdown("<p>a <strong>bold</strong> and <em>italic</em> line</p>"),
            "a bold and italic line"
        );
    }

    #[test]
    fn whitespace_between_elements_becomes_one_space() {
        assert_eq!(
            html_to_markdown("<p>\n   one\n   two   <span> three </span>\n</p>"),
            "one two three"
        );
    }

    #[test]
    fn a_line_break_breaks_the_line_without_ending_the_block() {
        assert_eq!(html_to_markdown("<p>first<br>second</p>"), "first\nsecond");
    }

    #[test]
    fn preformatted_text_keeps_its_shape() {
        assert_eq!(
            html_to_markdown("<html><pre>fn main() {\n    ok();\n}</pre></html>"),
            "fn main() {\n    ok();\n}"
        );
    }

    #[test]
    fn character_references_are_resolved() {
        assert_eq!(
            html_to_markdown("<p>a &amp; b &lt;c&gt; &quot;d&quot; &#39;e&#39; &#x41;</p>"),
            "a & b <c> \"d\" 'e' A"
        );
    }

    #[test]
    fn an_unknown_reference_is_left_alone() {
        assert_eq!(
            html_to_markdown("<p>Q&A and &notareference; stay</p>"),
            "Q&A and &notareference; stay"
        );
    }

    #[test]
    fn comments_and_doctypes_contribute_nothing() {
        assert_eq!(
            html_to_markdown("<!doctype html><!-- <p>hidden</p> --><p>shown</p>"),
            "shown"
        );
    }

    #[test]
    fn table_cells_are_separated() {
        assert_eq!(
            html_to_markdown("<html><table><tr><td>a</td><td>b</td></tr></table></html>"),
            "a b"
        );
    }

    #[test]
    fn a_tag_that_never_closes_does_not_lose_the_page() {
        // The page is malformed; what it says still reaches the model.
        assert_eq!(html_to_markdown("<p>before</p><div class=\"x"), "before");
        assert_eq!(html_to_markdown("<p>5 < 6 is true</p>"), "5 < 6 is true");
    }

    #[test]
    fn an_unterminated_script_swallows_only_the_rest_of_the_page() {
        assert_eq!(html_to_markdown("<p>before</p><script>alert(1)"), "before");
    }

    /// A page may close its script or style in any case, and may put
    /// whitespace before the `>`. Only the element's own closing tag ends it.
    #[test]
    fn a_closing_tag_in_another_case_still_ends_the_script_or_style() {
        assert_eq!(
            html_to_markdown("<SCRIPT>alert(1)</SCRIPT><p>Content</p>"),
            "Content"
        );
        assert_eq!(
            html_to_markdown("<Style>a{}</STYLE ><p>Content</p>"),
            "Content"
        );
    }

    #[test]
    fn a_closing_tag_of_another_element_is_script_text() {
        assert_eq!(
            html_to_markdown("<script>var s = \"</div>\";</script><p>Content</p>"),
            "Content"
        );
    }

    #[test]
    fn adjacent_and_nested_scripts_and_styles_are_all_dropped() {
        assert_eq!(
            html_to_markdown("<html><SCRIPT>a</SCRIPT><STYLE>b</STYLE><p>Content</p></html>"),
            "Content"
        );
        assert_eq!(
            html_to_markdown("<script>x<style>y</style>z</script><p>Content</p>"),
            "Content"
        );
    }

    #[test]
    fn an_unterminated_uppercase_script_swallows_only_the_rest_of_the_page() {
        assert_eq!(html_to_markdown("<p>before</p><SCRIPT>alert(1)"), "before");
    }

    /// A page that carries its real destination in `href` and a tracking copy
    /// in `data-href` must still link where it says it does.
    #[test]
    fn an_attribute_name_inside_another_one_is_not_the_attribute() {
        assert_eq!(
            html_to_markdown(
                r#"<p><a data-href="https://tracker.example" href="https://example.com">link</a></p>"#
            ),
            "[link](https://example.com)"
        );
    }

    /// The other half of the same rule: an `href` written inside another
    /// attribute's value is that value's text. A page that carries markup in a
    /// tracking or templating attribute must still link where the tag says it
    /// does.
    #[test]
    fn an_attribute_name_inside_a_quoted_value_is_not_the_attribute() {
        assert_eq!(
            html_to_markdown(
                r#"<p><a title='x href="https://tracker.example"' href="https://example.com">link</a></p>"#
            ),
            "[link](https://example.com)"
        );
    }

    #[test]
    fn attributes_are_read_case_insensitively_and_may_be_single_quoted() {
        assert_eq!(
            html_to_markdown("<p><A HREF='https://example.com/a?b=1&amp;c=2'>link</A></p>"),
            "[link](https://example.com/a?b=1&c=2)"
        );
    }
}
