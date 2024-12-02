import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from queue import Queue
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Website Content Scraper", 0, 1, "C")
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
    
    def add_content(self, title, content):
        self.add_page()
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, 0, 1)
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, content)

def scrape_website(base_url, pdf):
    visited = set()
    q = Queue()
    q.put(base_url)

    while not q.empty():
        url = q.get()
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
        pdf.add_content(url, page_text)

        for link in soup.find_all('a', href=True):
            full_url = urljoin(base_url, link['href'])
            if is_same_domain(base_url, full_url) and full_url not in visited:
                q.put(full_url)

def is_same_domain(base_url, target_url):
    base_netloc = urlparse(base_url).netloc
    target_netloc = urlparse(target_url).netloc
    return base_netloc == target_netloc

# Main Execution
def main():
    base_url = 'https://example.com'  # Replace with your target URL
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    scrape_website(base_url, pdf)
    pdf.output("scraped_content.pdf")
    print("PDF saved as 'scraped_content.pdf'")

if __name__ == "__main__":
    main()