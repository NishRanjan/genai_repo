import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Initialize the LLM for text generation (Hugging Face GPT-2)
llm = pipeline('text-generation', model='gpt2')

# Step 1: Scrape Udemy courses based on keywords
def scrape_udemy_courses(keywords):
    query = keywords.replace(" ", "+")
    url = f"https://www.udemy.com/courses/search/?src=ukw&q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    courses = soup.find_all('div', class_='course-card--container--1QM2W')
    st.write(courses)
    
    course_list = []
    for course in courses:
        title = course.find('div', class_='udlite-focus-visible-target udlite-heading-md course-card--course-title--2f7tE').text
        description = course.find('p', class_='udlite-text-sm course-card--course-headline--yIrRk').text
        course_list.append({"title": title, "description": description})
    
    return course_list

# Step 2: Use LLM to categorize courses by level (beginner, intermediate, advanced)
def categorize_courses_with_llm(courses):
    categorized_courses = {"Beginner": [], "Intermediate": [], "Advanced": []}
    
    # Use the LLM to categorize each course based on its description
    for course in courses:
        description = course['description']
        prompt = f"Categorize this course based on its difficulty level as 'beginner', 'intermediate', or 'advanced': {description}"
        
        generated_text = llm(prompt, max_length=50, num_return_sequences=1)[0]['generated_text'].lower()
        
        if 'beginner' in generated_text:
            categorized_courses['Beginner'].append(course)
        elif 'intermediate' in generated_text:
            categorized_courses['Intermediate'].append(course)
        elif 'advanced' in generated_text:
            categorized_courses['Advanced'].append(course)
    
    return categorized_courses

# Step 3: Generate a structured learning path from beginner to advanced
def generate_learning_path(categorized_courses):
    learning_path = []
    
    # Start with beginner courses
    if categorized_courses['Beginner']:
        learning_path.append("Beginner Level Courses:")
        for course in categorized_courses['Beginner']:
            learning_path.append(f"- {course['title']} - {course['description']}")
    
    # Add intermediate courses
    if categorized_courses['Intermediate']:
        learning_path.append("\nIntermediate Level Courses:")
        for course in categorized_courses['Intermediate']:
            learning_path.append(f"- {course['title']} - {course['description']}")
    
    # Finish with advanced courses
    if categorized_courses['Advanced']:
        learning_path.append("\nAdvanced Level Courses:")
        for course in categorized_courses['Advanced']:
            learning_path.append(f"- {course['title']} - {course['description']}")
    
    return "\n".join(learning_path)

# Streamlit App
def main():
    st.title("Personalized Learning Path Generator")
    
    # User input for the topic they want to learn
    user_query = st.text_input("Enter a topic you want to learn:", "")
    
    if st.button("Generate Learning Path"):
        if user_query:
            # Step 1: Scrape Udemy courses based on user input
            st.write("Scraping Udemy courses...")
            scraped_courses = scrape_udemy_courses(user_query)
            
            # Step 2: Categorize courses using LLM
            st.write("Categorizing courses with LLM...")
            categorized_courses = categorize_courses_with_llm(scraped_courses)
            
            # Step 3: Generate a learning path from beginner to advanced
            st.write("Generating learning path...")
            learning_path = generate_learning_path(categorized_courses)
            
            # Display the recommended learning path
            st.subheader("Recommended Learning Path:")
            st.text(learning_path)
        else:
            st.write("Please enter a valid learning topic.")
    
if __name__ == "__main__":
    main()
