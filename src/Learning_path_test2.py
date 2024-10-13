import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# Step 1: Set up Selenium WebDriver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode for efficiency
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Ensure to provide the correct path to your downloaded ChromeDriver
    service = Service(executable_path='chromedriver.exe')  # Update the path to chromedriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Step 2: Scrape Udemy courses using Selenium and BeautifulSoup
def scrape_udemy_courses(keyword):
    driver = setup_driver()
    # Use the updated URL structure as provided
    search_url = f"https://www.udemy.com/courses/search/?src=ukw&q={keyword.replace(' ', '+')}"
    driver.get(search_url)

    # Allow time for the page to load fully
    time.sleep(3)

    # Get the page source and parse it using BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Find all course elements based on the class name you provided
    courses = soup.find_all('h3', class_='ud-heading-md course-card-title-module--course-title--wmFXN')

    course_list = []
    for course in courses:
        try:
            title = course.find('a').text.strip()
            url = "https://www.udemy.com" + course.find('a')['href']
            
            # Find other SEO metadata such as ratings, reviews, and price
            metadata = course.find_next('div', class_='ud-sr-only')
            rating = metadata.find('span', {'data-testid': 'seo-rating'}).text.strip().replace('Rating: ', '')
            reviews = metadata.find('span', {'data-testid': 'seo-num-reviews'}).text.strip()
            price = metadata.find('span', {'data-testid': 'seo-current-price'}).text.strip().replace('Current price: ', '')

            course_list.append({
                "title": title,
                "url": url,
                "rating": rating,
                "reviews": reviews,
                "price": price
            })
        except AttributeError:
            continue

    driver.quit()
    return course_list

# Step 3: Generate the learning path (without LLM for now)
def generate_learning_path(courses):
    learning_path = []
    if courses:
        learning_path.append("Courses Found:\n")
        for course in courses:
            learning_path.append(f"- {course['title']} (Rating: {course['rating']}, Reviews: {course['reviews']}, Price: {course['price']})")
            learning_path.append(f"  [Course Link]({course['url']})\n")
    return "\n".join(learning_path)

# Streamlit App
def main():
    st.title("Udemy Course Scraper with Selenium")

    # User input for the topic they want to learn
    user_query = st.text_input("Enter a topic you want to learn:", "")

    if st.button("Search"):
        if user_query:
            st.write("Fetching courses from Udemy...")

            # Scrape Udemy courses using Selenium
            scraped_courses = scrape_udemy_courses(user_query)

            if scraped_courses:
                # Generate a learning path from the courses found
                learning_path = generate_learning_path(scraped_courses)

                # Display the learning path
                st.subheader("Courses Found:")
                st.markdown(learning_path, unsafe_allow_html=True)
            else:
                st.write("No courses found for the specified topic.")
        else:
            st.write("Please enter a valid learning topic.")

if __name__ == "__main__":
    main()
