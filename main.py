import os
import json
import asyncio
import re
import random # Import the random module
from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image, ImageDraw, ImageFont # Import more from Pillow

# --- Environment Variable Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# HUGGING_FACE_API_KEY is no longer needed

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pitch Deck PDF Generator & Viewer API (Public)",
    description="A public API to generate, save, and view pitch decks from audio or text.",
    version="8.0.0" # Final Hackathon Version with Template Images
)

# --- CORS Configuration ---
origins = ["http://localhost", "http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Client Initialization ---
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Pydantic Models & Helper Functions ---
class PitchDeckSlide(BaseModel):
    title: str
    content: str
class PitchDeckContent(BaseModel):
    company_name: str
    tagline: str
    slides: List[PitchDeckSlide]

def create_pitch_deck_prompt(idea_text: str) -> str:
    return f"""
    You are an expert startup consultant. Based on the user's idea: "{idea_text}", generate the content for a 5-slide pitch deck.
    Return a single, valid JSON object with the structure: {{"company_name": "...", "tagline": "...", "slides": [{{"title": "...", "content": "..."}}]}}.
    The 5 slides MUST be: 1. The Problem, 2. Our Solution, 3. Business Model, 4. Traction & Milestones, 5. The Ask.
    """
async def _generate_deck_content(idea_text: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = create_pitch_deck_prompt(idea_text)
    try:
        response = await model.generate_content_async(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON object found in the AI's response.")
        json_string = match.group(0)
        return json.loads(json_string)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI content generation failed: {e}")

async def _create_slide_image(slide_title: str) -> bytes:
    """Creates a simple image with a random background color and the slide title."""
    try:
        # Define image size
        width, height = 800, 400
        # Generate a pleasant random background color
        bg_color = (random.randint(100, 220), random.randint(100, 220), random.randint(100, 220))
        
        # Create a new image with the random color
        image = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(image)

        # Select a font
        try:
            # Use a common font if available
            font = ImageFont.truetype("arial.ttf", 50)
        except IOError:
            # Use a default font if the specific one isn't found
            font = ImageFont.load_default()

        # Calculate text position for centering
        text_bbox = draw.textbbox((0, 0), slide_title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((width - text_width) / 2, (height - text_height) / 2)

        # Draw the text on the image
        draw.text(position, slide_title, fill=(255, 255, 255), font=font)

        # Save image to an in-memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"Failed to create template image: {e}")
        return b'' # Return empty bytes on failure

def _create_pitch_deck_pdf(deck_content: dict, images: list) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(width / 2.0, height - 150, deck_content['company_name'])
    c.setFont("Helvetica", 18)
    c.drawCentredString(width / 2.0, height - 200, deck_content['tagline'])
    c.showPage()
    for i, slide in enumerate(deck_content['slides']):
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 100, slide['title'])
        p = Paragraph(slide['content'], styles['Normal'])
        p.wrapOn(c, width - 100, height - 450)
        p.drawOn(c, 50, height - 350)
        if i < len(images) and images[i]:
            try:
                image_reader = ImageReader(io.BytesIO(images[i]))
                c.drawImage(image_reader, 50, 80, width=width-100, height=200, preserveAspectRatio=True, anchor='n')
            except Exception as e:
                print(f"Could not draw image for slide {i+1}: {e}")
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

async def _process_and_save_idea(final_idea_text: str):
    try:
        print("Step 2: Generating deck text content...")
        deck_content = await _generate_deck_content(final_idea_text)
        print("Step 2 Complete.")

        print("Step 3: Creating template images for slides...")
        # Create an image for each slide using its title
        image_tasks = [_create_slide_image(slide['title']) for slide in deck_content['slides']]
        images = await asyncio.gather(*image_tasks)
        print("Step 3 Complete.")

        print("Step 4: Creating PDF document...")
        pdf_buffer = _create_pitch_deck_pdf(deck_content, images)
        pdf_bytes = pdf_buffer.getvalue()
        print("Step 4 Complete.")
        
        print("Step 5: Uploading PDF to Supabase Storage...")
        file_path = f"public/{uuid.uuid4()}.pdf"
        supabase.storage.from_("pitch-decks").upload(file=pdf_bytes, path=file_path, file_options={"content-type": "application/pdf"})
        pdf_url = supabase.storage.from_("pitch-decks").get_public_url(file_path)
        print("Step 5 Complete.")

        print("Step 6: Saving metadata to database...")
        deck_to_save = {
            "company_name": deck_content.get("company_name"),
            "tagline": deck_content.get("tagline"),
            "original_idea": final_idea_text,
            "pdf_url": pdf_url,
            "storage_path": file_path
        }
        db_response = supabase.table("pitch-decks").insert(deck_to_save).execute()
        print("Step 6 Complete.")
        
        return db_response.data[0]
    except Exception as e:
        print(f"\n--- ERROR DURING DECK GENERATION ---")
        print(f"An unexpected error occurred: {e}")
        print("--- END OF ERROR ---")
        raise HTTPException(status_code=500, detail=str(e))

# --- API Endpoints (No Authentication) ---

@app.post("/decks/text", tags=["Pitch Decks"])
async def generate_from_text(idea_text: str = Form(...)):
    print("\n--- Starting New Deck Generation from TEXT ---")
    if len(idea_text) < 20:
        raise HTTPException(status_code=400, detail="Text idea is too short.")
    new_deck_data = await _process_and_save_idea(idea_text)
    print("--- Deck Generation Successful ---")
    return JSONResponse(status_code=201, content=new_deck_data)

@app.post("/decks/audio", tags=["Pitch Decks"])
async def generate_from_audio(audio_file: UploadFile = File(...)):
    print("\n--- Starting New Deck Generation from AUDIO ---")
    print("Step 1: Transcribing audio with Gemini...")
    transcription_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    audio_bytes = await audio_file.read()
    audio_part = {"mime_type": audio_file.content_type, "data": audio_bytes}
    response = await transcription_model.generate_content_async(["Transcribe this startup idea.", audio_part])
    final_idea_text = response.text.strip()
    print(f"Step 1 Complete. Transcript: {final_idea_text[:100]}...")
    new_deck_data = await _process_and_save_idea(final_idea_text)
    print("--- Deck Generation Successful ---")
    return JSONResponse(status_code=201, content=new_deck_data)

@app.get("/decks", tags=["Pitch Decks"])
async def get_all_decks():
    response = supabase.table("pitch-decks").select("*").order("created_at", desc=True).execute()
    return response.data

@app.delete("/decks/{deck_id}", tags=["Pitch Decks"])
async def delete_deck(deck_id: int):
    select_res = supabase.table("pitch-decks").select("storage_path").eq("id", deck_id).single().execute()
    if not select_res.data:
        raise HTTPException(status_code=404, detail="Deck not found.")
    storage_path = select_res.data[0]['storage_path']
    supabase.storage.from_("pitch-decks").remove([storage_path])
    supabase.table("pitch-decks").delete().eq("id", deck_id).execute()
    return {"status": "success", "message": f"Deck {deck_id} deleted."}
