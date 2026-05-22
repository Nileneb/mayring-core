"""Identity layer — kanonische Workspace-Auflösung + lokales Login-Cache.

Module:
    workspace_resolver — resolve_workspace(input, default_user_id) -> str
    local_identity     — get/save lokal gecachte User-ID + Token

Architektur-Entscheidung (2026-05-09): workspace_id ist kein freier
String mehr, sondern wird gegen die `workspaces`-Tabelle in
cache/memory.db validiert / aufgelöst.

Auth-Quellen für CLI-Pfade (Reihenfolge):
    1. expliziter --workspace-Parameter (kann Alias oder kanonischer
       Workspace sein)
    2. ENV MAYRING_USER_ID (für non-interactive container)
    3. ~/.config/mayring/identity.json (von `mayring login` erstellt)
    4. raise → kein "default"-Fallback mehr im Auth-relevanten Pfad
"""
