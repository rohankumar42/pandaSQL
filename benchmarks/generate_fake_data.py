import csv
import argparse
import random

from faker import Faker
from faker.providers import isbn, company, person, address, lorem, date_time

fake = Faker()
fake.add_provider(isbn)
fake.add_provider(company)
fake.add_provider(person)
fake.add_provider(address)
fake.add_provider(lorem)
fake.add_provider(date_time)


def generate_book_and_authors(num_authors, num_books, append=False):
    mode = 'a' if append else 'w'

    authors = []
    with open('authors.csv', mode) as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'first_name',
            'last_name',
            'birth_day',
            'birth_month',
            'birth_year',
            'bio',
            'country',
        ])
        writer.writeheader()

        for _ in range(num_authors):
            dob = fake.date_of_birth(minimum_age=21)
            name = {
                'first_name': fake.first_name(),
                'last_name': fake.last_name()
            }
            author = {
                'first_name': name['first_name'],
                'last_name': name['last_name'],
                'birth_day': dob.day,
                'birth_month': dob.month,
                'birth_year': dob.year,
                'bio': fake.paragraph(),
                'country': fake.country(),
            }
            writer.writerow(author)

            authors.append(name)

    with open('books.csv', mode) as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'title',
            'publication_year',
            'ISBN10',
            'ISBN13',
            'keywords',
            'description',
            'first_name',
            'last_name',
        ])
        writer.writeheader()

        for _ in range(num_books):
            author = random.choice(authors)
            book = {
                'title': fake.bs().title(),
                'publication_year': fake.date_this_century().year,
                'ISBN10': fake.isbn10(),
                'ISBN13': fake.isbn13(),
                'keywords': fake.catch_phrase(),
                'description': fake.paragraph(),
                'first_name': author['first_name'],
                'last_name': author['last_name'],
            }
            writer.writerow(book)


def generate_top_authors(percent=0.01, seed=1):
    random.seed(seed)

    with open('authors.csv') as r_fh, open('top_authors.csv', 'w') as w_fh:
        reader = csv.DictReader(r_fh)
        writer = csv.DictWriter(w_fh, fieldnames=['first_name', 'last_name'])
        writer.writeheader()

        for row in reader:
            output = {
                'first_name': row['first_name'],
                'last_name': row['last_name']
            }
            if random.random() < percent:
                writer.writerow(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-authors', type=int, required=True)
    parser.add_argument('--num-books', type=int, required=True)
    parser.add_argument('--append', action='store_true', default=False)
    args = parser.parse_args()

    generate_book_and_authors(args.num_authors, args.num_books, args.append)
