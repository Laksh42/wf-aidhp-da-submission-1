use financial_advisor;
db.createCollection("users");
db.users.insertOne({username: "test", email: "test@example.com"});
db.getCollectionNames();
