from mayring_core.identity import authz


def test_roles_and_permissions_defined():
    assert authz.ROLES == ("user", "editor", "admin")
    assert set(authz.PERMISSIONS) == {"search", "write", "share_org", "share_public", "manage_members", "manage_foreign"}


def test_default_matrix_three_tier():
    d = authz.DEFAULT_MATRIX
    assert d[("user", "search")] is True
    assert d[("user", "write")] is True
    assert d[("user", "share_org")] is False
    assert d[("user", "manage_members")] is False
    assert d[("editor", "share_org")] is True
    assert d[("editor", "share_public")] is True
    assert d[("editor", "manage_members")] is False
    assert all(d[("admin", p)] is True for p in authz.PERMISSIONS)


def test_normalize_role_maps_legacy_vocab():
    assert authz.normalize_role("owner") == "admin"
    assert authz.normalize_role("viewer") == "user"
    assert authz.normalize_role("editor") == "editor"
    assert authz.normalize_role("admin") == "admin"
    assert authz.normalize_role("user") == "user"
    assert authz.normalize_role(None) == "user"
    assert authz.normalize_role("weird") == "user"
