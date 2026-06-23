"""Official class ordering and conversions between training and submission labels."""

from __future__ import annotations

# Bu dosyada resmî sınıf sırasını tek bir yerde sabit tutuyorum.
CLASSES = (
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
    "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
    "Golden_Retriever", "German_Shepherd", "Siberian_Husky",
    "Dalmatian", "Rottweiler",
)

# Eğitimde -1 kullanamadığım için reject sınıfını geçici olarak output 20 yapıyorum.
REJECT_EXTERNAL = -1
REJECT_INTERNAL = len(CLASSES)
NUM_OUTPUTS = len(CLASSES) + 1


def external_to_internal(label: int) -> int:
    """Convert submission reject label -1 into the trainable output index 20."""
    if label == REJECT_EXTERNAL:
        return REJECT_INTERNAL
    if 0 <= label < len(CLASSES):
        return label
    raise ValueError(f"Unsupported external label: {label}")


def internal_to_external(label: int) -> int:
    """Convert the trainable reject index 20 back into submission label -1."""
    if label == REJECT_INTERNAL:
        return REJECT_EXTERNAL
    if 0 <= label < len(CLASSES):
        return label
    raise ValueError(f"Unsupported internal label: {label}")
