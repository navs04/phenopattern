from app.search import MIN_QUERY_CHARS, SearchIndex, normalise


def test_normalise_collapses_punctuation_and_case():
    assert normalise("Bilateral Tonic-Clonic  seizure!") == "bilateral tonic clonic seizure"


def test_exact_beats_prefix_beats_word_beats_substring(store):
    hits = store.search_index.search("seizure", 10)
    names = [h.name for h in hits]
    assert names[0] == "Seizure"                                   # exact name
    assert names.index("Focal-onset seizure") < len(names)          # word match
    assert {"Focal-onset seizure", "Bilateral tonic-clonic seizure"} <= set(names)


def test_hyphenated_terms_are_found_by_spaced_words(store):
    assert store.search_index.search("tonic clonic", 5)[0].name == "Bilateral tonic-clonic seizure"


def test_synonym_match_reports_the_synonym_but_returns_the_official_name(store):
    hit = store.search_index.search("spider fing", 5)[0]
    assert (hit.id, hit.name, hit.matched_synonym) == ("HP:0001166", "Arachnodactyly", "Spider fingers")
    hit = store.search_index.search("arachno", 5)[0]
    assert hit.matched_synonym is None                              # official-name match


def test_official_name_outranks_a_synonym_of_the_same_quality(store):
    # "Seizure" matches its own name exactly; "Seizures" is only a synonym
    hits = store.search_index.search("seizures", 5)
    assert hits[0].id == "HP:0001250" and hits[0].matched_synonym == "Seizures"


def test_one_hit_per_term_and_limit(store):
    hits = store.search_index.search("seizure", 25)
    assert len({h.id for h in hits}) == len(hits)
    assert len(store.search_index.search("seizure", 1)) == 1


def test_short_empty_and_nonsense_queries(store):
    assert MIN_QUERY_CHARS == 2
    assert store.search_index.search("s") == [] and store.search_index.search("   ") == []
    assert store.search_index.search("zzzzqqq") == []
    assert store.search_index.search("seizure", 0) == []


def test_generic_terms_are_not_suggestible(store):
    assert not any(h.id in ("HP:0000001", "HP:0000118") for h in store.search_index.search("abnormality", 25))
    assert all(h.id != "HP:0000118" for h in store.search_index.search("phenotypic abnormality", 5))


def test_injection_like_input_is_just_text(store):
    assert store.search_index.search("'; drop table--", 5) == []
    assert store.search_index.search("<script>alert(1)</script>", 5) == []


def test_ordering_is_deterministic():
    idx = SearchIndex([("HP:2", "Aa x", []), ("HP:1", "Aa y", [])])
    assert [h.id for h in idx.search("aa")] == ["HP:2", "HP:1"]       # same tier and length -> alphabetical by name
    assert idx.search("aa") == idx.search("aa")
