RESOURCE_PRODUCERS = {
    "NtOpenFile": ("file", "FileHandle"),
    "NtCreateFile": ("file", "FileHandle"),

    "NtOpenKey": ("registry", "KeyHandle"),
    "NtOpenKeyEx": ("registry", "KeyHandle"),
    "RegOpenKeyExA": ("registry", "Handle"),
    "RegOpenKeyExW": ("registry", "Handle"),

    "NtCreateSection": ("section", "SectionHandle"),
    "NtOpenSection": ("section", "SectionHandle"),

    "NtCreateMutant": ("mutant", "Handle"),
    "NtOpenEvent": ("event", "Handle"),

    "NtOpenProcessToken": ("token", None),

    "NtCreateUserProcess": ("process", None),
    "CreateProcessW": ("process", None),

    "CoCreateInstance": ("com", None),
    "HeapCreate": ("heap", None),
}


RESOURCE_RELEASES = {
    "NtClose": "Handle",
    "RegCloseKey": "Handle",
    "NtReleaseMutant": "Handle",
}


def extract_acquisition(event):
    """
    Extract a resource acquisition from one normalized CAPE event.

    Returns:
        dict | None
    """
    api = event.get("api")

    if api not in RESOURCE_PRODUCERS:
        return None

    # Failed API calls do not create resources.
    if event.get("status") is False:
        return None

    resource_type, handle_arg = RESOURCE_PRODUCERS[api]

    arguments = event.get("arguments") or event.get("args") or []

    # APIs such as HeapCreate, CoCreateInstance, and process creation
    # do not expose the resulting resource handle in a simple argument.
    if handle_arg is None:
        return {
            "api": api,
            "resource_type": resource_type,
            "handle": None,
            "sequence": event.get("sequence"),
            "timestamp": event.get("timestamp"),
            "arguments": arguments,
            "process_id": event.get("process_id"),
            "event_id": event.get("id"),
        }

    handle = None

    for arg in arguments:
        if arg.get("name") == handle_arg:
            handle = arg.get("value")
            break

    if not handle or handle == "0x00000000":
        return None

    return {
        "api": api,
        "resource_type": resource_type,
        "handle": handle,
        "sequence": event.get("sequence"),
        "timestamp": event.get("timestamp"),
        "arguments": arguments,
        "process_id": event.get("process_id"),
        "event_id": event.get("id"),
    }


def extract_release(event):
    """
    Extract a resource release from one normalized CAPE event.

    Returns:
        dict | None
    """
    api = event.get("api")

    if api not in RESOURCE_RELEASES:
        return None

    handle_arg = RESOURCE_RELEASES[api]

    arguments = event.get("arguments") or event.get("args") or []

    handle = None

    for arg in arguments:
        if arg.get("name") == handle_arg:
            handle = arg.get("value")
            break

    if not handle or handle == "0x00000000":
        return None

    return {
        "api": api,
        "handle": handle,
        "sequence": event.get("sequence"),
        "timestamp": event.get("timestamp"),
        "process_id": event.get("process_id"),
        "event_id": event.get("id"),
    }