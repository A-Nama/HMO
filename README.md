# Hear Me Out

## Basic Details

- **Team Name:** barbie.dev  
- **Team Members:** Anjana Sankar  
- **Track:** Media  
- **Problem Statement:**  
  Great ideas often strike us at unexpected moments—while walking, showering, or chatting with a friend. But these ideas are frequently forgotten before they’re captured or developed. There’s no lightweight, accessible solution to quickly record and structure them into something actionable.

- **Solution:**  
  Hear Me Out is a GenAI-powered tool that helps you instantly document voice or text ideas and turn them into structured pitch decks. It ensures your fleeting thoughts aren’t lost but preserved in a form you can present, share, or develop further.

- **Project Description:**  
  Users can either record their idea via voice or type it in. The backend uses Gemini AI to transcribe, interpret, and expand the idea into a structured 10-slide pitch deck. This is then converted into a presentation file and uploaded to Supabase, with a public link and metadata generated and stored.

---

## Technical Details

- **Tech Stack and Libraries Used:**
  - **Backend:** FastAPI (Python), Google Generative AI (Gemini Pro – Text & Audio), Supabase (Database & Storage)
  - **Frontend (Prototype):** Basic HTML + Tailwind CSS
  - **Database Table:** `pitch_decks`  
    - Fields: `company_name`, `tagline`, `original_idea`, `pdf_url`, `storage_path`

- **Implementation:**  
  1. User inputs an idea (via audio or text).
  2. Gemini AI transcribes audio and expands the content into a 10-slide pitch.
  3. The generated deck is converted into a PPT file.
  4. The file is uploaded to Supabase Storage.
  5. A public link is returned, and metadata is stored in the database.

---

## Installation and Execution Instructions

```bash
git clone https://github.com/yourusername/hearmeout
cd hearmeout
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Screenshots

![WhatsApp Image 2025-07-20 at 12 27 29](https://github.com/user-attachments/assets/c51f41a2-f230-4c7d-b2e9-409bd81e6c4e)
![WhatsApp Image 2025-07-20 at 12 26 54](https://github.com/user-attachments/assets/f9249863-e757-43a8-a8b2-e7f69f042a7a)
![WhatsApp Image 2025-07-20 at 12 28 00](https://github.com/user-attachments/assets/3009644d-404c-462f-936e-0a33c375452c)



## Project Demo

Watch a walkthrough of how **Hear Me Out** works:  
[▶️ Demo Video](https://drive.google.com/file/d/1f08eJhdZdkl7CmpXGVltFbmtL5Ci2Cad/view?usp=sharing)

---

**Made with love ❤️ by barbie.dev**
