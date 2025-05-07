import os
import csv
import random

# --- Phishing Email Templates and Components ---
phishing_subjects_starters = [
    "Urgent Security Alert:", "Action Required:", "Account Verification Needed:", "Unusual Login Attempt Detected:",
    "Your Account Has Been Suspended:", "Important Notification Regarding Your Account:", "You Have a Pending Refund:",
    "Invoice Due:", "Password Expiration Notice:", "Confirm Your Details Immediately:"
]
phishing_greetings = [
    "Dear Valued Customer,", "Dear User,", "Hello,", "Greetings,", "Dear Account Holder,"
]
phishing_body_templates = [
    "We detected unusual activity on your account from an unrecognized device. Please verify your identity immediately by clicking the link below to avoid suspension: {link}",
    "Your account password is set to expire in 24 hours. To maintain access, please update your credentials here: {link}",
    "Our records indicate your account information is outdated. Failure to update within 48 hours will result in service disruption. Update now: {link}",
    "You have a pending payment of ${amount}. Please settle this immediately to avoid late fees. View invoice and pay: {link}",
    "Congratulations! You've been selected to receive a ${prize_value} gift card. Claim your reward now: {link}",
    "A recent login attempt was made to your account from IP address {ip_address}. If this was not you, please secure your account immediately: {link}",
    "Important security update for your {service_name} account. Please review and confirm your settings: {link}",
    "There is an important document shared with you. Please login to view: {link}. Your login is your email address.",
    "We've noticed some suspicious activity linked to your payment method. Please re-verify your billing details to prevent a hold on your account: {link}",
    "Your mailbox is almost full. To avoid losing incoming emails, please upgrade your storage via this link: {link}"
]
phishing_closings = [
    "Sincerely,", "Regards,", "Best,", "The {company_name} Security Team", "Customer Support"
]
phishing_links = [
    "secure-login-portal.com/auth", "account-update-verification.net/validate", "service-provider-confirm.org/signin",
    "payment-notification-center.com/details", "reward-center-claims.info/redeem", "myservice-alert-resolution.com/act",
    "official-security-update.net/user", "document-share-access.com/login", "billing-secure-verify.org/submit", "webmail-upgrade-portal.com/expand"
]
phishing_companies = [
    "PayPaI", "Microsofft", "Amaz0n", "Netfliz", "YourBank", "Googgle", "Appple", "WellFargo", "CitiiBank", "USPSecure"
]

# --- Safe Email Templates and Components ---
safe_subjects_starters = [
    "Your Order Confirmation #{order_number}", "Weekly Newsletter:", "Meeting Reminder:", "Project Update:",
    "Thank You for Your Purchase!", "Your Monthly Statement is Ready", "Welcome to {service_name}!",
    "Following Up On Our Conversation", "Quick Question About {topic}", "Scheduled Maintenance Notification"
]
safe_greetings = [
    "Hi {name},", "Hello {name},", "Dear {name},", "Team,", "Hi everyone,"
]
safe_body_templates = [
    "Just a friendly reminder about our meeting scheduled for {date} at {time} regarding {meeting_topic}. Please find the agenda attached.",
    "Thanks for your recent order #{order_number}! We're preparing it for shipment and will notify you once it's on its way. You can track your order here: {link}",
    "Welcome to the {service_name} family! We're thrilled to have you. Here are some resources to get you started: {link}",
    "This week's newsletter is packed with exciting updates, including {feature1} and {feature2}. Read more here: {link}",
    "This is a notification that our services will undergo scheduled maintenance on {date} from {start_time} to {end_time}. We apologize for any inconvenience.",
    "Following up on our discussion about {project_name}, I've attached the revised proposal for your review. Let me know your thoughts.",
    "Your monthly e-statement for account ending in {account_digits} is now available. Please log in to your secure portal to view it: {link}",
    "I hope this email finds you well. I wanted to share an interesting article I found on {topic}: {link}",
    "Thank you for contacting customer support. We have received your query (Ticket ID: {ticket_id}) and a representative will get back to you shortly.",
    "The team has made significant progress on the {project_name} initiative. Key updates include: {update1}, {update2}. More details can be found on our internal portal: {link}"
]
safe_closings = [
    "Best regards,", "Sincerely,", "Thanks,", "All the best,", "Regards,"
]
safe_links = [
    "official-company-website.com/orders", "our-brand.com/newsletter", "internal-meeting-tool.com/join",
    "legit-service-provider.com/welcome", "status.ourservice.com/maintenance", "project-management-portal.com/docs",
    "secure.bankname.com/statements", "trusted-news-source.com/article", "support.mycompany.com/tickets", "company-intranet.com/updates"
]
safe_names = ["Alex", "Jamie", "Chris", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Devin", "Drew"]
safe_service_names = ["ConnectSphere", "InnovateHub", "DataStream", "CloudCorp", "MarketMind"]

def generate_random_string(length=10):
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))

def generate_email_body(is_phishing):
    body = ""
    if is_phishing:
        subject_starter = random.choice(phishing_subjects_starters)
        greeting = random.choice(phishing_greetings)
        template = random.choice(phishing_body_templates)
        closing = random.choice(phishing_closings).format(company_name=random.choice(phishing_companies))
        link = "http://" + random.choice(phishing_links) + "/" + generate_random_string(8)
        company = random.choice(phishing_companies)

        body_content = template.format(
            link=link,
            amount=random.randint(20, 500),
            prize_value=random.randint(50, 1000),
            ip_address=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            service_name=company
        )
        # Add some misspellings or urgency markers subtly
        if random.random() < 0.3:
            body_content = body_content.replace(" a ", " an ", 1).replace("is", "si", 1) # subtle typo
        if random.random() < 0.4:
            body_content += " Act fast, this offer is limited!"

        body = f"Subject: {subject_starter} {company}\n\n{greeting}\n\n{body_content}\n\n{closing}\n{company} Support"

    else: # Safe email
        name = random.choice(safe_names)
        subject_starter = random.choice(safe_subjects_starters).format(
            order_number=random.randint(10000, 99999),
            service_name=random.choice(safe_service_names),
            topic=random.choice(["our recent call", "the Q2 report", "next week's agenda"])
        )
        greeting = random.choice(safe_greetings).format(name=name)
        template = random.choice(safe_body_templates)
        closing = random.choice(safe_closings)
        link = "https://" + random.choice(safe_links) + "/" + generate_random_string(6)
        service = random.choice(safe_service_names)

        body_content = template.format(
            name=name,
            order_number=random.randint(10000, 99999),
            link=link,
            date=f"May {random.randint(10,28)}, 2025",
            time=f"{random.randint(1,12)}:{random.choice(['00','15','30','45'])} {random.choice(['AM','PM'])}",
            meeting_topic=random.choice(["Q3 Planning", "Product Feedback", "Client Onboarding"]),
            service_name=service,
            feature1=random.choice(["new analytics dashboard", "enhanced security features", "mobile app improvements"]),
            feature2=random.choice(["upcoming webinars", "user success stories", "integration options"]),
            start_time=f"{random.randint(1,5)}:00 AM UTC",
            end_time=f"{random.randint(6,10)}:00 AM UTC",
            project_name=random.choice(["Alpha Launch", "Beta Test Program", "Client Portal Upgrade"]),
            account_digits=random.randint(1000,9999),
            topic=random.choice(["industry trends", "new technology", "competitive analysis"]),
            ticket_id=generate_random_string(8).upper(),
            update1=random.choice(["user interface redesigned", "database migration complete", "API endpoints tested"]),
            update2=random.choice(["documentation updated", "performance benchmarks exceeded", "user feedback collected"])
        )
        body = f"Subject: {subject_starter}\n\n{greeting}\n\n{body_content}\n\n{closing}\n{name if random.random() > 0.5 else service + ' Team'}"

    # Ensure varying lengths
    if len(body) < 100 and random.random() < 0.7: # Shorter emails sometimes get padded
        padding_sentences = [
            "Please let me know if you have any questions.", "Feel free to reach out if you need further assistance.",
            "We appreciate your attention to this matter.", "Thank you for your cooperation.",
            "This is an automated message, please do not reply directly.", "For more information, visit our website."
            "If you believe you received this in error, please contact support immediately."
        ]
        body += "\n\n" + random.choice(padding_sentences)
        if random.random() < 0.3:
             body += " " + random.choice(padding_sentences) # Add more padding

    elif len(body) > 700 and random.random() < 0.5: # Longer emails sometimes get truncated
        body = body[:random.randint(650,700)] + "..."


    # Add more random text to vary length further if needed
    if random.random() < 0.2:
        body += "\n\nPS: " + " ".join([generate_random_string(random.randint(3,8)) for _ in range(random.randint(5,15))])
    if random.random() < 0.2 and is_phishing:
        body += "\n\nRef: " + generate_random_string(12)
    if random.random() < 0.2 and not is_phishing:
        body += "\n\nAttachment: " + random.choice(["summary.pdf", "details.docx", "agenda.pptx", "image_screenshot.png"])


    return body.strip()

# --- Generate Dataset ---
emails_data = []
num_emails_per_category = 500

for _ in range(num_emails_per_category):
    emails_data.append({"body": generate_email_body(is_phishing=True), "label": 1})

for _ in range(num_emails_per_category):
    emails_data.append({"body": generate_email_body(is_phishing=False), "label": 0})

random.shuffle(emails_data) # Shuffle the dataset

# --- Write to CSV ---
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)
csv_file = os.path.join(out_dir, "emails.csv")
csv_columns = ["body", "label"]

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in emails_data:
            writer.writerow(data)
    print(f"Successfully generated {csv_file} with {len(emails_data)} emails.")
except IOError:
    print("I/O error")