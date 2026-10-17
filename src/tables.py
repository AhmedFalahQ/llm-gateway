""" Local mirrors of two DynamoDB tables"""

from __future__ import annotations

KEY_SCHEMA = [
    {"AttributeName": "pk", "KeyType": "HASH"},
    {"AttributeName": "sk", "KeyType": "RANGE"},
]
ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "pk", "AttributeType": "S"},
    {"AttributeName": "sk", "AttributeType": "S"},
]

def create_table(resource, name: str):
    """Create one gateway table on a boto3 DynamoDB resource."""
    return resource.create_table(
        TableName=name,
        KeySchema=KEY_SCHEMA,
        AttributeDefinitions=ATTRIBUTE_DEFINITIONS,
        BillingMode="PAY_PER_REQUEST",
    )


def create_local_tables(
    resource, budget_table: str = "test-budgets", cache_table: str = "test-cache"
):
    """Create both tables and return them as `(budget, cache)`."""
    return create_table(resource, budget_table), create_table(resource, cache_table)
