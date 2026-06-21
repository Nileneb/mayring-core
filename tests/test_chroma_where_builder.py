"""Tenancy Phase A (T5) + repo-scoping-hardfilter + reference-doc-layer.

The Chroma $where mirrors _scope_filter so public/org chunks from other tenants
come back as vector candidates. workspace_id is no longer a hard filter. On top
of the visibility disjunction it now hard-filters by repo (repo-scoping) and
default-excludes source_class='reference' (reference-doc-layer).
"""
from mayring_core.memory.retrieval import build_chroma_where


def _visibility_part(where: dict) -> dict:
    """Extract the visibility disjunction from a possibly $and-wrapped where."""
    if "$and" in where:
        # First element is always the visibility clause.
        return where["$and"][0]
    return where


def _extra_clauses(where: dict) -> list:
    return where["$and"][1:] if "$and" in where else []


class TestBuildChromaWhere:

    def test_member_disjunction_has_public_private_and_org_in(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=("org-x",)
        )
        vis = _visibility_part(where)
        assert "$or" in vis
        ors = vis["$or"]

        assert {"visibility": "public"} in ors

        private_clause = {"$and": [{"visibility": "private"}, {"user_id": "u-me"}]}
        assert private_clause in ors

        org_clauses = [
            o for o in ors
            if isinstance(o, dict) and o.get("$and")
            and {"visibility": "org"} in o["$and"]
        ]
        assert len(org_clauses) == 1
        org_and = org_clauses[0]["$and"]
        assert {"org_id": {"$in": ["org-x"]}} in org_and

    def test_no_org_membership_omits_org_clause_no_empty_in(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=()
        )
        vis = _visibility_part(where)
        # No org membership → no org clause at all, never $in: [].
        flat = repr(vis)
        assert '"$in"' not in flat and "'$in'" not in flat
        assert "org" not in flat

        ors = vis["$or"]
        assert {"visibility": "public"} in ors
        assert {"$and": [{"visibility": "private"}, {"user_id": "u-me"}]} in ors

    def test_public_only_visibility_when_no_user_no_orgs(self):
        where = build_chroma_where(workspace_id="ws-a", user_id=None, org_ids=())
        # Single visibility clause, but still wrapped with the default
        # reference exclusion.
        assert _visibility_part(where) == {"visibility": "public"}

    # --- reference-doc-layer ------------------------------------------------

    def test_default_excludes_reference(self):
        where = build_chroma_where(workspace_id="ws-a", user_id="u-me", org_ids=())
        assert {"source_class": {"$ne": "reference"}} in _extra_clauses(where)

    def test_include_reference_omits_exclusion(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=(), include_reference=True
        )
        flat = repr(where)
        assert "source_class" not in flat

    def test_reference_only_filters_to_reference(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=(), reference_only=True
        )
        assert {"source_class": "reference"} in _extra_clauses(where)

    # --- repo-scoping-hardfilter -------------------------------------------

    def test_repo_adds_hard_filter(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=(), repo="owner/app"
        )
        assert {"repo": "owner/app"} in _extra_clauses(where)

    def test_repo_and_reference_compose(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=(), repo="owner/app"
        )
        extra = _extra_clauses(where)
        assert {"repo": "owner/app"} in extra
        assert {"source_class": {"$ne": "reference"}} in extra
