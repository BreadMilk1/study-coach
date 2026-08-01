"""Upload size limits shared by route-level PDF checks and ASGI body gating.

`MAX_UPLOAD_BYTES` is the exact PDF file content limit enforced while streaming
the UploadFile to a temp path (second-layer defense).

`MAX_UPLOAD_REQUEST_BYTES` caps the entire multipart HTTP body *before*
Starlette's multipart parser spools parts. It equals the file limit plus a
bounded overhead allowance for multipart boundaries and part headers.
"""

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_UPLOAD_REQUEST_BYTES = MAX_UPLOAD_BYTES + MAX_UPLOAD_MULTIPART_OVERHEAD_BYTES
