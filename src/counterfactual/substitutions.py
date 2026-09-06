"""Behavioral substitution library for the counterfactual search.

Each entry encodes a plausible behavioral-equivalence claim: the listed
substitutes achieve the same high-level goal (persistence, injection, …)
via a different mechanism.  The library is hand-curated and cross-checked
against published malware-family reports and the MITRE ATT&CK technique
descriptions.  Provenance strings are included for traceability in results
and for the analyst-facing diff view.

Design notes
------------
* Substitutions are **one-level**: A ↔ B means A can be replaced by B and
  vice-versa.  Multi-step techniques (e.g. process-hollowing) are NOT
  represented as single API nodes; such compound patterns must be handled
  at the graph level, not here.
* Resource names are kept semantically correct for each target API.  A
  registry-key path is not a valid scheduled-task name, so we map resource
  strings to a neutral placeholder that the downstream classifier can treat
  as a resource of the target type rather than a corrupted value.
* The library currently covers ~28 canonical pairs across five groups.
  Expanding it further (target: 40–50 pairs for the corpus evaluation) is
  the single highest-value next step for improving substitution coverage.
"""
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Resource-type hint: maps an API name to the kind of resource it operates on.
# Used by update_resources_for_substitution to produce a plausible resource
# string for the substituted API rather than a mangled version of the original.
# ---------------------------------------------------------------------------
_API_RESOURCE_TYPE: Dict[str, str] = {
    # Persistence – registry
    "RegSetValue":         "registry_key",
    "NtSetValueKey":       "registry_key",
    "ZwSetValueKey":       "registry_key",
    # Persistence – scheduled task
    "CreateScheduledTask": "scheduled_task",
    "ITaskScheduler":      "scheduled_task",
    # Persistence – service
    "CreateService":       "service_name",
    "OpenSCManagerW":      "service_name",
    # Persistence – startup folder
    "CopyFileW":           "file_path",
    "MoveFileW":           "file_path",
    # Injection – thread
    "CreateRemoteThread":  "remote_thread",
    "CreateRemoteThreadEx":"remote_thread",
    "NtQueueApcThread":    "apc_thread",
    "NtQueueApcThreadEx":  "apc_thread",
    "RtlCreateUserThread": "remote_thread",
    # Injection – memory
    "NtWriteVirtualMemory":"virtual_memory",
    "WriteProcessMemory":  "virtual_memory",
    # Injection – mapping
    "NtMapViewOfSection":  "mapped_section",
    "MapViewOfFile":       "mapped_section",
    # File write
    "NtWriteFile":         "file_path",
    "WriteFile":           "file_path",
    # Network/C2
    "WinHttpOpen":         "http_session",
    "InternetOpenA":       "http_session",
    "InternetOpenW":       "http_session",
    "socket":              "network_socket",
    "WSASocketW":          "network_socket",
    "connect":             "network_connection",
    "WSAConnect":          "network_connection",
    # Memory
    "VirtualAllocEx":      "virtual_alloc",
    "NtAllocateVirtualMemory": "virtual_alloc",
}

_RESOURCE_PLACEHOLDER: Dict[str, str] = {
    "registry_key":      "HKCU\\Software\\[substituted]",
    "scheduled_task":    "\\Microsoft\\Windows\\[substituted]",
    "service_name":      "[SubstitutedService]",
    "file_path":         "C:\\Users\\Public\\[substituted]",
    "remote_thread":     "[remote_thread_handle]",
    "apc_thread":        "[apc_thread_handle]",
    "virtual_memory":    "[virtual_memory_region]",
    "mapped_section":    "[mapped_section_handle]",
    "http_session":      "[http_session_handle]",
    "network_socket":    "[socket_handle]",
    "network_connection":"[connection_handle]",
    "virtual_alloc":     "[virtual_alloc_region]",
}


def update_resources_for_substitution(
    api_name: str,
    original_resources: Optional[List[str]] = None,
) -> List[str]:
    """Return a resource list appropriate for the substituted API.

    Rather than naively prefixing the original resource strings (which
    produces semantically invalid values such as
    ``createscheduledtask:HKLM\\...``), we look up the *resource type* of
    the target API and return a neutral placeholder of that type.  If the
    original resource list is empty, we return a single placeholder so the
    substituted node still has a resource slot.
    """
    resource_type = _API_RESOURCE_TYPE.get(api_name)
    if resource_type is None:
        # Unknown API: fall back to a single generic placeholder rather than
        # mangling the original values.
        return [f"{api_name.lower()}:[resource]"] if api_name else []

    placeholder = _RESOURCE_PLACEHOLDER.get(resource_type, f"[{resource_type}]")

    if not original_resources:
        return [placeholder]

    # Preserve the count of resource slots but replace their content with
    # type-appropriate placeholders so the classifier sees a plausible
    # resource list for the substituted API.
    return [placeholder] * len(original_resources)


# ---------------------------------------------------------------------------
# Substitution library
# Groups: persistence, injection, file-write, network/C2, memory-manipulation
# ---------------------------------------------------------------------------
SUBSTITUTION_LIBRARY: Dict[str, List[str]] = {
    # ── Persistence: registry ↔ scheduled-task ↔ service ──────────────────
    "RegSetValue":          ["CreateScheduledTask", "CreateService"],
    "CreateService":        ["RegSetValue", "CreateScheduledTask"],
    "CreateScheduledTask":  ["RegSetValue", "CreateService"],

    # ── Persistence: startup-folder file copy (lateral to the above) ───────
    # CopyFileW / MoveFileW to the Startup folder achieves registry-equivalent
    # auto-run persistence on login (T1547.001).
    "CopyFileW":            ["RegSetValue", "CreateScheduledTask"],
    "MoveFileW":            ["RegSetValue", "CreateScheduledTask"],

    # ── Code injection: remote thread ↔ APC ↔ user-thread ─────────────────
    # All three are single-call injection primitives observable in sandbox
    # traces.  CreateRemoteThreadEx is the extended form of CreateRemoteThread
    # and is treated as its alias here.
    "CreateRemoteThread":   ["NtQueueApcThread", "RtlCreateUserThread"],
    "CreateRemoteThreadEx": ["NtQueueApcThread", "RtlCreateUserThread"],
    "NtQueueApcThread":     ["CreateRemoteThread", "RtlCreateUserThread"],
    "NtQueueApcThreadEx":   ["CreateRemoteThread", "RtlCreateUserThread"],
    "RtlCreateUserThread":  ["CreateRemoteThread", "NtQueueApcThread"],

    # ── Memory write: user-space WriteProcessMemory ↔ NT-layer NtWrite ─────
    "WriteProcessMemory":   ["NtWriteVirtualMemory"],
    "NtWriteVirtualMemory": ["WriteProcessMemory"],

    # ── Section mapping: Win32 MapViewOfFile ↔ NT NtMapViewOfSection ────────
    "MapViewOfFile":        ["NtMapViewOfSection"],
    "NtMapViewOfSection":   ["MapViewOfFile"],

    # ── Virtual allocation: VirtualAllocEx ↔ NtAllocateVirtualMemory ────────
    "VirtualAllocEx":       ["NtAllocateVirtualMemory"],
    "NtAllocateVirtualMemory": ["VirtualAllocEx"],

    # ── File write: Win32 WriteFile ↔ NT NtWriteFile ────────────────────────
    "WriteFile":            ["NtWriteFile"],
    "NtWriteFile":          ["WriteFile"],

    # ── Network/C2: WinHTTP ↔ WinINet (both achieve HTTP C2) ────────────────
    "WinHttpOpen":          ["InternetOpenA", "InternetOpenW"],
    "InternetOpenA":        ["WinHttpOpen", "InternetOpenW"],
    "InternetOpenW":        ["WinHttpOpen", "InternetOpenA"],

    # ── Network/C2: raw socket API variants ─────────────────────────────────
    "socket":               ["WSASocketW"],
    "WSASocketW":           ["socket"],
    "connect":              ["WSAConnect"],
    "WSAConnect":           ["connect"],
}


# ---------------------------------------------------------------------------
# Provenance: (original, substitute) → short human-readable rationale
# ---------------------------------------------------------------------------
SUBSTITUTION_PROVENANCE: Dict[Tuple[str, str], str] = {
    ("RegSetValue", "CreateScheduledTask"):  "registry run-key ↔ scheduled-task persistence (T1547/T1053)",
    ("RegSetValue", "CreateService"):        "registry run-key ↔ service-based persistence (T1547/T1543)",
    ("CreateService", "RegSetValue"):        "service install can be replaced by registry run-key",
    ("CreateService", "CreateScheduledTask"):"service ↔ scheduled-task persistence (both T1543/T1053)",
    ("CreateScheduledTask", "RegSetValue"):  "scheduled-task ↔ registry run-key (T1053/T1547)",
    ("CreateScheduledTask", "CreateService"):"scheduled-task ↔ service install (T1053/T1543)",
    ("CopyFileW", "RegSetValue"):            "startup-folder copy ↔ registry run-key (both T1547)",
    ("CopyFileW", "CreateScheduledTask"):    "startup-folder copy ↔ scheduled-task (T1547/T1053)",
    ("MoveFileW", "RegSetValue"):            "startup-folder move ↔ registry run-key (both T1547)",
    ("MoveFileW", "CreateScheduledTask"):    "startup-folder move ↔ scheduled-task (T1547/T1053)",
    ("CreateRemoteThread", "NtQueueApcThread"):      "remote-thread ↔ APC injection (T1055.003/T1055.004)",
    ("CreateRemoteThread", "RtlCreateUserThread"):   "remote-thread ↔ user-thread injection (T1055.003)",
    ("CreateRemoteThreadEx", "NtQueueApcThread"):    "extended remote-thread ↔ APC injection",
    ("CreateRemoteThreadEx", "RtlCreateUserThread"): "extended remote-thread ↔ user-thread injection",
    ("NtQueueApcThread", "CreateRemoteThread"):      "APC ↔ remote-thread injection (T1055.004/T1055.003)",
    ("NtQueueApcThread", "RtlCreateUserThread"):     "APC ↔ user-thread injection",
    ("NtQueueApcThreadEx", "CreateRemoteThread"):    "extended APC ↔ remote-thread injection",
    ("NtQueueApcThreadEx", "RtlCreateUserThread"):   "extended APC ↔ user-thread injection",
    ("RtlCreateUserThread", "CreateRemoteThread"):   "user-thread ↔ remote-thread injection",
    ("RtlCreateUserThread", "NtQueueApcThread"):     "user-thread ↔ APC injection",
    ("WriteProcessMemory", "NtWriteVirtualMemory"):  "Win32 ↔ NT-layer memory write (same syscall path)",
    ("NtWriteVirtualMemory", "WriteProcessMemory"):  "NT-layer ↔ Win32 memory write",
    ("MapViewOfFile", "NtMapViewOfSection"):         "Win32 ↔ NT-layer section map (T1055.001)",
    ("NtMapViewOfSection", "MapViewOfFile"):         "NT-layer ↔ Win32 section map",
    ("VirtualAllocEx", "NtAllocateVirtualMemory"):   "Win32 ↔ NT-layer virtual alloc (T1055)",
    ("NtAllocateVirtualMemory", "VirtualAllocEx"):   "NT-layer ↔ Win32 virtual alloc",
    ("WriteFile", "NtWriteFile"):                    "Win32 ↔ NT-layer file write",
    ("NtWriteFile", "WriteFile"):                    "NT-layer ↔ Win32 file write",
    ("WinHttpOpen", "InternetOpenA"):                "WinHTTP ↔ WinINet HTTP session (T1071.001)",
    ("WinHttpOpen", "InternetOpenW"):                "WinHTTP ↔ WinINet HTTP session (unicode) (T1071.001)",
    ("InternetOpenA", "WinHttpOpen"):                "WinINet ↔ WinHTTP HTTP session",
    ("InternetOpenA", "InternetOpenW"):              "ANSI ↔ Unicode WinINet open",
    ("InternetOpenW", "WinHttpOpen"):                "WinINet (unicode) ↔ WinHTTP session",
    ("InternetOpenW", "InternetOpenA"):              "Unicode ↔ ANSI WinINet open",
    ("socket", "WSASocketW"):                        "POSIX socket ↔ Winsock2 WSASocketW (T1095)",
    ("WSASocketW", "socket"):                        "Winsock2 ↔ POSIX socket",
    ("connect", "WSAConnect"):                       "POSIX connect ↔ Winsock2 WSAConnect (T1095)",
    ("WSAConnect", "connect"):                       "Winsock2 WSAConnect ↔ POSIX connect",
}


# ---------------------------------------------------------------------------
# Alias table: maps lowercase API-name variants to the canonical key used in
# SUBSTITUTION_LIBRARY.  This replaces the old substring-scan approach.
# ---------------------------------------------------------------------------
API_ALIASES: Dict[str, str] = {
    # RegSetValue variants
    "regsetvalueexa":         "RegSetValue",
    "regsetvalueexw":         "RegSetValue",
    "zwsetvaluekey":          "RegSetValue",
    "ntsetvaluekey":          "RegSetValue",
    "regsetvaluea":           "RegSetValue",
    "regsetvaluew":           "RegSetValue",
    # CreateRemoteThread variants
    "createremotethreadex":   "CreateRemoteThread",
    # NtQueueApcThread variants
    "ntqueueapcthreadex":     "NtQueueApcThreadEx",
    "zwqueueapcthread":       "NtQueueApcThread",
    # WriteProcessMemory (no common variant)
    # NtWriteVirtualMemory
    "zwwritevirtualmemory":   "NtWriteVirtualMemory",
    # MapViewOfFile
    "mapviewoffileex":        "MapViewOfFile",
    # NtMapViewOfSection
    "zwmapviewofsection":     "NtMapViewOfSection",
    # VirtualAllocEx
    "virtualallocexnuma":     "VirtualAllocEx",
    # NtAllocateVirtualMemory
    "zwallocatevirtualmemory":"NtAllocateVirtualMemory",
    # WriteFile
    "writefileex":            "WriteFile",
    # NtWriteFile
    "zwwritefile":            "NtWriteFile",
    # WinHttp
    "winhttpopenw":           "WinHttpOpen",
    # InternetOpen
    # socket aliases
    "wsasocketa":             "WSASocketW",
    # connect aliases
    "connectex":              "connect",
}


def get_substitutes(api_name: str) -> List[str]:
    """Return the list of substitute API names for *api_name*.

    Lookup order:
    1. Exact match on SUBSTITUTION_LIBRARY keys (case-sensitive, as stored).
    2. Lowercase-exact match on SUBSTITUTION_LIBRARY keys.
    3. Alias resolution via API_ALIASES (lowercase → canonical key).
    Returns [] if no substitutes are known.
    """
    if api_name is None:
        return []
    if api_name in SUBSTITUTION_LIBRARY:
        return SUBSTITUTION_LIBRARY[api_name]
    normalized = api_name.lower()
    # Case-insensitive exact match
    for key in SUBSTITUTION_LIBRARY:
        if normalized == key.lower():
            return SUBSTITUTION_LIBRARY[key]
    # Alias resolution
    canonical = API_ALIASES.get(normalized)
    if canonical and canonical in SUBSTITUTION_LIBRARY:
        return SUBSTITUTION_LIBRARY[canonical]
    return []


def get_substitution_provenance(original: str, substitute: str) -> Optional[str]:
    """Return the provenance comment for a substitution pair, or None."""
    return SUBSTITUTION_PROVENANCE.get((original, substitute))
