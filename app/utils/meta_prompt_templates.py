"""
Meta-prompt templates for generating personalized financial advice
These templates are used by the meta_prompt_generator to create 
personalized prompts for the financial wellness assistant
"""

USER_PROFILE_TEMPLATE = """
# USER PROFILE INFORMATION
Name: {name}
Age: {age}
Occupation: {occupation}
Location: {location}
Annual Income: ${annual_income:,.2f}
Annual Expenses: ${annual_expenses:,.2f}
Risk Tolerance: {risk_tolerance}
Financial Goals: {financial_goals}
"""

FINANCIAL_CONTEXT_TEMPLATE = """
# FINANCIAL CONTEXT
## Account Information
Account Type: {account_type}
Checking Balance: ${checking_balance:,.2f}
Savings Balance: ${savings_balance:,.2f}
Total Balance: ${total_balance:,.2f}

## Credit Information
Credit Score: {credit_score}
Outstanding Debt: ${outstanding_debt:,.2f}
Credit Utilization: {credit_utilization}%
Payment History: {payment_history}
Credit Age: {credit_age_years} years

## Investment Portfolio
{investment_summary}
"""

TRANSACTION_CONTEXT_TEMPLATE = """
# RECENT TRANSACTIONS (Last 30 days)
{recent_transactions}

## Transaction Insights
Largest Expense: {largest_expense}
Most Frequent Category: {most_frequent_category}
Unusual Activity: {unusual_activity}
"""

INVESTMENT_SUMMARY_TEMPLATE = """
Investment Type: {investment_type}
Current Value: ${current_value:,.2f}
Initial Investment: ${initial_amount:,.2f}
Growth/Loss: {growth_loss}
Start Date: {start_date}
"""

TRANSACTION_ITEM_TEMPLATE = """
Date: {date}
Amount: ${amount:,.2f}
Category: {category}
Merchant: {merchant}
"""

SOCIAL_MEDIA_TEMPLATE = """
# SOCIAL MEDIA INSIGHTS
Recent Topics: {topics}
Overall Sentiment: {overall_sentiment}
Financial Interests: {financial_interests}
"""

FULL_META_PROMPT_TEMPLATE = """
You are a personalized financial wellness assistant for {name}, a {age}-year-old {occupation} based in {location}. When providing recommendations and advice, consider the following comprehensive profile:

{user_profile}

{financial_context}

{transaction_context}

{social_media_context}

# YOUR ROLE AS A FINANCIAL ASSISTANT
Always maintain a helpful, supportive tone while providing personalized financial advice based on the above information. Tailor your recommendations specifically to {name}'s financial situation, goals, and behavioral patterns.

When responding:
1. Address {name} directly and personally
2. Provide advice that aligns with their risk tolerance ({risk_tolerance})
3. Focus on their stated financial goals: {financial_goals}
4. Reference their specific financial situation and recent activity
5. Make specific, actionable recommendations that match their life circumstances
6. Explain financial concepts in clear language appropriate for someone with {name}'s background
7. Prioritize long-term financial wellness while addressing immediate concerns

Avoid generic advice that doesn't account for {name}'s unique situation.
"""

def format_financial_goals(goals_list):
    """Format a list of financial goals into a readable string"""
    if not goals_list:
        return "No specific financial goals provided"
    
    return ", ".join(goals_list)

def format_transaction_summary(transactions, limit=5):
    """Format a summary of recent transactions"""
    if not transactions:
        return "No recent transactions"
    
    formatted_transactions = []
    for i, txn in enumerate(transactions[:limit]):
        formatted_txn = TRANSACTION_ITEM_TEMPLATE.format(
            date=txn.get('date', 'Unknown date'),
            amount=float(txn.get('amount', 0)),
            category=txn.get('category', 'Uncategorized'),
            merchant=txn.get('merchant', 'Unknown merchant')
        )
        formatted_transactions.append(formatted_txn)
    
    if len(transactions) > limit:
        formatted_transactions.append(f"\n...and {len(transactions) - limit} more transactions")
    
    return "\n\n".join(formatted_transactions)

def format_investment_summary(investments):
    """Format a summary of investment holdings"""
    if not investments:
        return "No current investments"
    
    total_value = sum(float(inv.get('current_value', 0)) for inv in investments)
    total_initial = sum(float(inv.get('amount', 0)) for inv in investments)
    
    # Group investments by type
    investments_by_type = {}
    for inv in investments:
        inv_type = inv.get('investment_type', 'Other')
        if inv_type not in investments_by_type:
            investments_by_type[inv_type] = []
        investments_by_type[inv_type].append(inv)
    
    summary_parts = [f"Total Portfolio Value: ${total_value:,.2f}"]
    
    for inv_type, invs in investments_by_type.items():
        type_value = sum(float(inv.get('current_value', 0)) for inv in invs)
        summary_parts.append(f"{inv_type}: ${type_value:,.2f} ({(type_value/total_value*100):.1f}%)")
    
    # Calculate overall growth/loss
    if total_initial > 0:
        growth_pct = (total_value - total_initial) / total_initial * 100
        growth_text = f"Overall Growth: {growth_pct:.1f}%"
        if growth_pct >= 0:
            growth_text += " (positive)"
        else:
            growth_text += " (negative)"
        summary_parts.append(growth_text)
    
    return "\n".join(summary_parts)

def format_social_media_insights(social_posts):
    """Format insights from social media data"""
    if not social_posts:
        return "No social media data available"
    
    # Extract topics from all posts
    all_topics = []
    for post in social_posts:
        if 'topics' in post and post['topics']:
            topics = post['topics'].split(',')
            all_topics.extend([t.strip() for t in topics])
    
    # Count topic frequency
    topic_counts = {}
    for topic in all_topics:
        if topic not in topic_counts:
            topic_counts[topic] = 0
        topic_counts[topic] += 1
    
    # Get most common topics
    common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    common_topics_str = ", ".join([t[0] for t in common_topics]) if common_topics else "None identified"
    
    # Calculate sentiment
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for post in social_posts:
        sentiment = post.get('sentiment', '').lower()
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    
    total_posts = len(social_posts)
    if total_posts > 0:
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        sentiment_ratio = f"{sentiment_counts['positive']}/{sentiment_counts['neutral']}/{sentiment_counts['negative']}"
    else:
        dominant_sentiment = "No data"
        sentiment_ratio = "N/A"
    
    # Extract financial interests
    financial_keywords = ["invest", "save", "budget", "finance", "money", "loan", "debt", "mortgage", 
                         "retirement", "stock", "bond", "fund", "bank", "credit", "tax"]
    
    financial_interests = []
    for post in social_posts:
        post_text = post.get('post_text', '').lower()
        for keyword in financial_keywords:
            if keyword in post_text and keyword not in financial_interests:
                financial_interests.append(keyword)
    
    financial_interests_str = ", ".join(financial_interests) if financial_interests else "None explicitly mentioned"
    
    return f"""
Primary Interest Areas: {common_topics_str}
Overall Sentiment: {dominant_sentiment.title()} (Positive/Neutral/Negative ratio: {sentiment_ratio})
Financial Topics: {financial_interests_str}
""" 