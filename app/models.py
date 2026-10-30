from sqlalchemy import Column, String, Text, Enum, JSON, ForeignKey, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from .db import Base
import enum

class ProjectStatus(str, enum.Enum):
    draft = "draft"
    generating = "generating"
    preview_ready = "preview_ready"
    awaiting_payment = "awaiting_payment"
    paid = "paid"
    generating_book = "generating_book"
    book_ready = "book_ready"
    book_failed = "book_failed"
    delivered = "delivered"

class Project(Base):
    __tablename__ = "projects"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    theme = Column(String, nullable=False)
    genre = Column(String, nullable=False, default="adventure")
    status = Column(Enum(ProjectStatus), nullable=False, default=ProjectStatus.draft)

    selected_story_option_id = Column(UUID(as_uuid=True), nullable=True)
    customer_email = Column(String, nullable=True)
    product_type = Column(String, nullable=True)        # "digital" | "print"
    panel_count = Column(Integer, nullable=True)        # 24 | 48 — panels generated for the full book
    shipping_address = Column(JSON, nullable=True)      # populated for print orders
    book_pdf_key = Column(String, nullable=True)        # digital (spread layout)
    book_print_pdf_key = Column(String, nullable=True)  # deprecated: old combined cover+interior PDF (Mixam)
    book_cover_pdf_key = Column(String, nullable=True)      # print production wrap cover (Lulu)
    book_interior_pdf_key = Column(String, nullable=True)   # print production interior pages (Lulu)
    lulu_print_job_id = Column(String, nullable=True)       # Lulu Print-Job id once submitted
    lulu_print_job_status = Column(String, nullable=True)   # last known status from Lulu (CREATED, IN_PRODUCTION, SHIPPED, ...)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Character(Base):
    __tablename__ = "characters"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="main")   # main | support | pet
    age = Column(Integer, nullable=True)                    # <-- make sure this line exists
    traits = Column(JSON, nullable=True)                    # appearance extracted from photo
    personalization = Column(JSON, nullable=True)           # customer-entered story personalization
    photo_keys = Column(JSON, nullable=True)
    reference_image_key = Column(String, nullable=True)
    reference_image_url = Column(String, nullable=True)
    reference_prompt = Column(Text, nullable=True)
    fal_lora_url = Column(String, nullable=True)
    fal_lora_status = Column(String, nullable=True)
    fal_training_request_id = Column(String, nullable=True)

class StoryOption(Base):
    __tablename__ = "story_options"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    logline = Column(String, nullable=False)
    outline = Column(JSON, nullable=False)
    estimated_pages = Column(Integer, nullable=False, default=24)

class FunnelEvent(Base):
    __tablename__ = "funnel_events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), nullable=True)
    event = Column(String, nullable=False)   # e.g. "preview_viewed", "options_viewed"
    meta = Column(JSON, nullable=True)       # optional extra context
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PreviewPage(Base):
    __tablename__ = "preview_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    story_option_id = Column(UUID(as_uuid=True), ForeignKey("story_options.id", ondelete="CASCADE"), nullable=False)

    page_number = Column(Integer, nullable=False)
    narration = Column(Text, nullable=False)
    dialogue = Column(JSON, nullable=True)      # list of dialogue lines is better than one long string
    image_prompt = Column(Text, nullable=False)
    image_url = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())