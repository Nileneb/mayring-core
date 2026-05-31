"""Tenancy Phase A (T5): build_chroma_where visibility disjunction.

The Chroma $where must mirror _scope_filter so public/org chunks from other
tenants come back as vector candidates. workspace_id is no longer a hard filter.
"""
from mayring_core.memory.retrieval import build_chroma_where


class TestBuildChromaWhere:

    def test_member_disjunction_has_public_private_and_org_in(self):
        where = build_chroma_where(
            workspace_id="ws-a", user_id="u-me", org_ids=("org-x",)
        )
        assert "$or" in where
        ors = where["$or"]

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
        # No org membership → no org clause at all, never $in: [].
        flat = repr(where)
        assert '"$in"' not in flat and "'$in'" not in flat
        assert "org" not in flat

        ors = where["$or"]
        assert {"visibility": "public"} in ors
        assert {"$and": [{"visibility": "private"}, {"user_id": "u-me"}]} in ors

    def test_public_only_when_no_user_no_orgs(self):
        where = build_chroma_where(workspace_id="ws-a", user_id=None, org_ids=())
        # Single clause → no $or wrapper.
        assert where == {"visibility": "public"}
