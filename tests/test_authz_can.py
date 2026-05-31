from mayring_core.identity.authz import can


def test_super_admin_bypass():
    assert can("user", "manage_members", is_super_admin=True) is True
    assert can("user", "manage_foreign", is_super_admin=True, overrides={}) is True


def test_default_matrix_applies_without_overrides():
    assert can("user", "write") is True
    assert can("user", "share_org") is False
    assert can("editor", "share_public") is True
    assert can("editor", "manage_members") is False
    assert can("admin", "manage_foreign") is True


def test_overrides_win_over_default():
    ov = {("editor", "share_public"): False}
    assert can("editor", "share_public", overrides=ov) is False
    assert can("editor", "share_org", overrides=ov) is True


def test_admin_locked_permission_cannot_be_revoked():
    ov = {("admin", "manage_members"): False}
    assert can("admin", "manage_members", overrides=ov) is True


def test_public_workspace_manage_members_super_admin_only():
    assert can("admin", "manage_members", is_public_ws=True) is False
    assert can("admin", "manage_members", is_public_ws=True, is_super_admin=True) is True
    assert can("editor", "share_public", is_public_ws=True) is True
