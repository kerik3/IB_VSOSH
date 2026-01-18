from app import app, db
from models import User, UserRole
from faker import Faker
import random

fake = Faker('ru_RU')

def seed_users(count=20):
    with app.app_context():
        print(f"Creating {count} fictitious students...")
        
        for i in range(count):
            # Create unique username
            base_username = fake.user_name()
            username = f"{base_username}_{random.randint(100, 999)}"
            
            while User.query.filter_by(username=username).first():
                username = f"{base_username}_{random.randint(100, 999)}"
            
            email = fake.email()
            while User.query.filter_by(email=email).first():
                email = fake.email()
            
            # Generate unique 8-digit ID
            while True:
                new_id = random.randint(10000000, 99999999)
                if not User.query.get(new_id):
                    break
            
            user = User(
                id=new_id,
                username=username,
                email=email,
                full_name=fake.name(),
                role=UserRole.STUDENT,
                is_active=True
            )
            user.set_password('student123')
            db.session.add(user)
            
        db.session.commit()
        print("Done!")

if __name__ == "__main__":
    seed_users(50)
