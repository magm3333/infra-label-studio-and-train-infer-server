"""Cifrado/descifrado de artifacts del Model Hub.

Formato ``.enc`` (ver ENCRYPTION.md en el repo models-hub):

    [magic "MHENC1" 6B][salt 16B][nonce 12B][ciphertext + tag GCM]

AES-256-GCM con clave derivada por scrypt de una passphrase. El salt es
aleatorio por archivo -> aunque se reutilice la misma passphrase, cada
archivo cifrado es distinto. El nonce nunca se reutiliza con la misma clave
derivada (se deriva una clave nueva -vía salt nuevo- en cada cifrado).
"""
import hashlib
import os
from pathlib import Path
from typing import Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"MHENC1"
SALT_LEN = 16
NONCE_LEN = 12
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32


class DecryptionError(Exception):
    """Clave incorrecta, formato inválido o archivo corrupto."""


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_bytes(data: bytes, passphrase: str) -> bytes:
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return MAGIC + salt + nonce + ciphertext


def decrypt_bytes(blob: bytes, passphrase: str) -> bytes:
    if len(blob) < len(MAGIC) + SALT_LEN + NONCE_LEN:
        raise DecryptionError("Archivo demasiado corto para ser un .enc válido")
    magic = blob[: len(MAGIC)]
    if magic != MAGIC:
        raise DecryptionError(f"Magic bytes inválidos: {magic!r} (esperado {MAGIC!r})")
    offset = len(MAGIC)
    salt = blob[offset : offset + SALT_LEN]
    offset += SALT_LEN
    nonce = blob[offset : offset + NONCE_LEN]
    offset += NONCE_LEN
    ciphertext = blob[offset:]
    key = _derive_key(passphrase, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as exc:  # InvalidTag u otro error de la librería
        raise DecryptionError(
            "No se pudo desencriptar: clave incorrecta o archivo corrupto"
        ) from exc


def encrypt_file(src: Union[str, Path], dest: Union[str, Path], passphrase: str) -> None:
    data = Path(src).read_bytes()
    Path(dest).write_bytes(encrypt_bytes(data, passphrase))


def decrypt_file(src: Union[str, Path], dest: Union[str, Path], passphrase: str) -> None:
    blob = Path(src).read_bytes()
    Path(dest).write_bytes(decrypt_bytes(blob, passphrase))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    return sha256_bytes(Path(path).read_bytes())
