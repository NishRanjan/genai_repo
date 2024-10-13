import streamlit as st
from udemyscraper import UdemyCourse
import cohere
import os

# Initialize the Cohere API (replace with your API key)
COHERE_API_KEY = os.getenv('COHERE_API_KEY')  # Replace with your actual key
co = cohere.Client(COHERE_API_KEY)

# Step 1: Fetch Udemy courses using udemyscraper
def fetch_udemy_courses(keyword):
    try:
        # Fetch courses based on the user query
        course = UdemyCourse()
        results = course.fetch_course(keyword)
        course_list = []
        
        # Process the fetched courses
        for res in results:
            course_info = {
                "title": res.get('title'),
                "url": res.get('url'),
                "price": res.get('price'),
                "description": res.get('description'),
                "rating": res.get('rating'),
                "level": res.get('level')
            }
            course_list.append(course_info)
        
        return course_list
    except Exception as e:
        st.error(f"Error fetching courses: {e}")
        return []

# Step 2: Use LLM (Cohere) to classify courses by level (beginner, intermediate, advanced)
def classify_courses_with_llm(courses):
    categorized_courses = {"Beginner": [], "Intermediate": [], "Advanced": []}
    
    for course in courses:
        description = course.get('description', '')
        prompt = f"Classify the following course as beginner, intermediate, or advanced: {description}"

        # Call Cohere's Generate API to classify the course
        response = co.generate(
            model='xlarge',
            prompt=prompt,
            max_tokens=50
        )

        generated_text = response.generations[0].text.strip().lower()

        if 'beginner' in generated_text:
            categorized_courses['Beginner'].append(course)
        elif 'intermediate' in generated_text:
            categorized_courses['Intermediate'].append(course)
        elif 'advanced' in generated_text:
            categorized_courses['Advanced'].append(course)
    
    return categorized_courses

# Step 3: Use LLM to generate a learning path explanation
def generate_learning_path_with_llm(categorized_courses):
    learning_path = []
    
    # Add beginner courses
    if categorized_courses['Beginner']:
        learning_path.append("Beginner Level Courses:")
        for course in categorized_courses['Beginner']:
            prompt = f"Why should a beginner start with this course? {course['description']}"
            response = co.generate(
                model='xlarge',
                prompt=prompt,
                max_tokens=100
            )
            explanation = response.generations[0].text.strip()
            learning_path.append(f"- {course['title']} (Rating: {course['rating']}, Price: {course['price']})\n  {course['description']} - [Link]({course['url']})\nExplanation: {explanation}")

    # Add intermediate courses
    if categorized_courses['Intermediate']:
        learning_path.append("\nIntermediate Level Courses:")
        for course in categorized_courses['Intermediate']:
            prompt = f"Why should an intermediate learner take this course? {course['description']}"
            response = co.generate(
                model='xlarge',
                prompt=prompt,
                max_tokens=100
            )
            explanation = response.generations[0].text.strip()
            learning_path.append(f"- {course['title']} (Rating: {course['rating']}, Price: {course['price']})\n  {course['description']} - [Link]({course['url']})\nExplanation: {explanation}")

    # Add advanced courses
    if categorized_courses['Advanced']:
        learning_path.append("\nAdvanced Level Courses:")
        for course in categorized_courses['Advanced']:
            prompt = f"Why should an advanced learner take this course? {course['description']}"
            response = co.generate(
                model='xlarge',
                prompt=prompt,
                max_tokens=100
            )
            explanation = response.generations[0].text.strip()
            learning_path.append(f"- {course['title']} (Rating: {course['rating']}, Price: {course['price']})\n  {course['description']} - [Link]({course['url']})\nExplanation: {explanation}")
    
    return "\n".join(learning_path)

# Streamlit App
def main():
    st.title("Personalized Learning Path Generator (Udemy Scraper + LLM)")

    # User input for the topic they want to learn
    user_query = st.text_input("Enter a topic you want to learn:", "")

    if st.button("Generate Learning Path"):
        if user_query:
            st.write("Fetching Udemy courses...")

            # Step 1: Fetch Udemy courses using udemyscraper
            scraped_courses = fetch_udemy_courses(user_query)

            if scraped_courses:
                # Step 2: Classify courses using LLM (Cohere)
                st.write("Classifying courses with LLM...")
                categorized_courses = classify_courses_with_llm(scraped_courses)

                # Step 3: Generate a learning path with LLM explanations
                st.write("Generating learning path with LLM...")
                learning_path = generate_learning_path_with_llm(categorized_courses)

                # Display the recommended learning path
                st.subheader("Recommended Learning Path:")
                st.markdown(learning_path, unsafe_allow_html=True)
            else:
                st.write("No courses found for the specified topic.")
        else:
            st.write("Please enter a valid learning topic.")

if __name__ == "__main__":
    main()
