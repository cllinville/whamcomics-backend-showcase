from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

VALID_GENRES = {"adventure", "mystery", "comedy", "fantasy", "sci_fi", "sports", "horror", "friendship", "horror_adult", "romance"}

class ProjectCreate(BaseModel):
    theme: str
    genre: str = "adventure"

class ProjectOut(BaseModel):
    id: uuid.UUID
    theme: str
    genre: str = "adventure"
    status: str
    selected_story_option_id: uuid.UUID | None = None

    class Config:
        from_attributes = True

class CharacterCreate(BaseModel):
    name: str
    role: str = "main"                 # "main" | "support" | "pet"
    age: Optional[int] = None
    traits: Optional[dict] = None
    photo_keys: Optional[List[str]] = None  # you can pass keys from /uploads/sign flow
    reference_image_url: Optional[str] = None
    reference_prompt: Optional[str] = None

class CharacterOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    role: str
    age: Optional[int] = None
    traits: Optional[dict] = None
    personalization: Optional[dict] = None
    photo_keys: Optional[List[str]] = None
    reference_image_key: str | None = None
    reference_image_url: Optional[str] = None
    reference_prompt: Optional[str] = None
    fal_lora_url: Optional[str] = None
    fal_lora_status: Optional[str] = None
    fal_training_request_id: Optional[str] = None

    class Config:
        from_attributes = True

class StoryOptionOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    logline: str
    outline: str
    estimated_pages: int

    class Config:
        from_attributes = True

class StoryChooseIn(BaseModel):
    story_option_id: uuid.UUID

class CharacterPersonalizationIn(BaseModel):
    age: Optional[int] = None                         # 7
    favorite_activities: Optional[List[str]] = None   # ["soccer", "drawing", "coding"]
    favorite_animal: Optional[str] = None             # "dogs"
    personality_traits: Optional[List[str]] = None    # ["brave", "funny", "creative"]
    favorite_place: Optional[str] = None              # "the beach"
    dream: Optional[str] = None                       # "become an astronaut"
    special_skill: Optional[str] = None               # "great at puzzles"
    character_gender: Optional[str] = None            # "boy" or "girl" — overrides AI extraction
    catchphrase: Optional[str] = None                 # "Let's go!"
    relationship_to_main: Optional[str] = None        # "husband", "sister", "coworker", etc. (support chars only)
    nickname_for_hero: Optional[str] = None           # what this support char calls the hero (e.g. "babe", "sis")
    hero_nickname_for_them: Optional[str] = None      # what the hero calls this support char (e.g. "Vince", "babe")

class CharacterPhotoAttach(BaseModel):
    add_keys: List[str]

class CharacterPhotoReplace(BaseModel):
    photo_keys: List[str]

VALID_PRODUCT_TYPES = {"digital", "print"}
VALID_PANEL_COUNTS = {24, 48}

class CheckoutRequest(BaseModel):
    project_id: uuid.UUID
    story_option_id: uuid.UUID | None = None  # falls back to project's selected option
    product_type: str = "digital"   # "digital" | "print"
    panel_count: int = 48           # 24 | 48

class DialogueLine(BaseModel):
    speaker: str
    line: str


class PreviewPageOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    story_option_id: uuid.UUID
    page_number: int
    narration: str
    dialogue: Optional[List[DialogueLine]] = None
    image_prompt: str
    image_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class PreviewPageCreate(BaseModel):
    page_number: int
    narration: str
    dialogue: Optional[List[DialogueLine]] = None
    image_prompt: str