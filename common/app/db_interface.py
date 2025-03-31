# db_interface.py

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
import psycopg2.extras  # Important for DictCursor

# --- Database Interface Class ---

class DatabaseInterface:
    """
    Provides an interface for interacting with the application database.
    Handles connection management and executes predefined queries.
    """
    def __init__(self, db_config: Dict[str, str]):
        """
        Initializes the DatabaseInterface with database configuration.

        Args:
            db_config (Dict[str, str]): Dictionary containing database connection details.
        """
        self.db_config = db_config
        print("DatabaseInterface initialized.")

    def _connect_db(self) -> Optional[psycopg2.extensions.connection]:
        """
        Internal method to establish a connection to the PostgreSQL database.

        Returns:
            Optional[psycopg2.extensions.connection]: A connection object or None if connection fails.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            # print("Database connection established successfully.")
            return conn
        except Exception as e:
            print(f"Database connection failed: {e}")
            return None

    def _execute_query(self, query: str, params: Optional[tuple] = None, fetch_one: bool = False, is_dict_cursor: bool = True) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any], Tuple, Any]]:
        """
        Internal helper method to execute a query and fetch results.

        Args:
            query (str): The SQL query string to execute.
            params (Optional[tuple]): Parameters to pass to the query (for preventing SQL injection).
            fetch_one (bool): If True, fetch only one row. Otherwise, fetch all.
            is_dict_cursor (bool): If True, use DictCursor to return results as dictionaries.

        Returns:
            Optional[Union[List[Dict[str, Any]], Dict[str, Any], Tuple, Any]]: Query results or None on error.
                 Format depends on fetch_one and is_dict_cursor.
        """
        conn = self._connect_db()
        if not conn:
            print("Cannot execute query due to connection failure.")
            return None

        results = None
        try:
            cursor_factory = psycopg2.extras.DictCursor if is_dict_cursor else None
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                cur.execute(query, params)
                if fetch_one:
                    results = cur.fetchone()
                else:
                    results = cur.fetchall()
                # print(f"Query executed successfully. Results: {results}") # Debugging
        except Exception as e:
            print(f"Query execution failed: {e}")
        finally:
            if conn:
                conn.close()
                # print("Database connection closed.")
        return results

    # --- User Table Functions ---

    def get_user_count(self) -> Optional[int]:
        """
        Fetch the total number of users in the database.

        Args:
            None

        Returns:
            Optional[int]: The total count of users, or None if an error occurs.
        """
        query = "SELECT COUNT(*) FROM users;"
        result = self._execute_query(query, fetch_one=True, is_dict_cursor=False)
        return result[0] if result else None # result will be a tuple like (count,)

    def get_user_details(self, user_id: Optional[int] = None, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve details for a specific user by ID or email.
        Provide either user_id or email.

        Args:
            user_id (Optional[int]): The ID of the user to retrieve.
            email (Optional[str]): The email address of the user to retrieve.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing user details (id, name, email, created_at)
                                      or None if not found or error.
        """
        if user_id is not None:
            query = "SELECT id, name, email, created_at FROM users WHERE id = %s;"
            params = (user_id,)
        elif email is not None:
            query = "SELECT id, name, email, created_at FROM users WHERE email = %s;"
            params = (email,)
        else:
            print("Error: Must provide either user_id or email.")
            return None

        result = self._execute_query(query, params, fetch_one=True, is_dict_cursor=True)
        return dict(result) if result else None # Convert DictRow to dict

    def get_users_by_creation_date(self, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve users created within a specific date range.

        Args:
            start_date (str): The start date (YYYY-MM-DD).
            end_date (str): The end date (YYYY-MM-DD).

        Returns:
            Optional[List[Dict[str, Any]]]: A list of user dictionaries, or None on error.
        """
        query = """
            SELECT id, name, email, created_at
            FROM users
            WHERE created_at::date BETWEEN %s AND %s
            ORDER BY created_at;
        """
        params = (start_date, end_date)
        results = self._execute_query(query, params, is_dict_cursor=True)
        return [dict(row) for row in results] if results is not None else None

    # --- Order Table Functions ---

    def get_orders_by_timeline(self, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a summary of items purchased within a specific time range.

        Args:
            start_date (str): The start date of the time range (YYYY-MM-DD).
            end_date (str): The end date of the time range (YYYY-MM-DD).

        Returns:
            Optional[List[Dict[str, Any]]]: A list of dictionaries, each containing
                                            'purchased_item' and 'count', or None on error.
        """
        # Assuming order_date includes time, casting to ::date for comparison
        query = """
            SELECT purchased_item, count(*)
            FROM orders
            WHERE order_date::date BETWEEN %s AND %s
            GROUP BY purchased_item
            ORDER BY purchased_item;
        """
        params = (start_date, end_date)
        results = self._execute_query(query, params, is_dict_cursor=True)
        # Convert DictRows to plain dictionaries
        return [dict(row) for row in results] if results is not None else None

    def get_order_status_count(self) -> Optional[Dict[str, int]]:
        """
        Get the count of orders grouped by their processing status.

        Args:
            None

        Returns:
            Optional[Dict[str, int]]: A dictionary mapping order status to its count,
                                      or None if an error occurs.
        """
        query = """
            SELECT order_status, COUNT(*) as count
            FROM orders
            GROUP BY order_status;
        """
        results = self._execute_query(query, is_dict_cursor=True)
        return {row["order_status"]: row["count"] for row in results} if results is not None else None

    def get_order_details(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve details for a specific order by its ID. Includes user email.

        Args:
            order_id (int): The ID of the order.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing order details, including user email,
                                      or None if not found or error.
        """
        query = """
            SELECT o.id, o.user_id, u.email as user_email, o.order_date, o.purchased_item, o.order_status, o.time_of_purchase
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE o.id = %s;
        """
        params = (order_id,)
        result = self._execute_query(query, params, fetch_one=True, is_dict_cursor=True)
        return dict(result) if result else None

    def get_orders_by_user(self, user_id: Optional[int] = None, email: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve all orders placed by a specific user (identified by ID or email).

        Args:
            user_id (Optional[int]): The ID of the user.
            email (Optional[str]): The email address of the user.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of order dictionaries for the user, or None on error.
        """
        if user_id is not None:
            query = """
                SELECT id, user_id, order_date, purchased_item, order_status, time_of_purchase
                FROM orders
                WHERE user_id = %s
                ORDER BY order_date DESC;
            """
            params = (user_id,)
        elif email is not None:
            # Find user_id from email first
            user_details = self.get_user_details(email=email)
            if not user_details:
                print(f"No user found with email: {email}")
                return None
            user_id = user_details.get('id') if isinstance(user_details, dict) else None
            query = """
                SELECT id, user_id, order_date, purchased_item, order_status, time_of_purchase
                FROM orders
                WHERE user_id = %s
                ORDER BY order_date DESC;
            """
            params = (user_id,)
        else:
            print("Error: Must provide either user_id or email.")
            return None

        results = self._execute_query(query, params, is_dict_cursor=True)
        return [dict(row) for row in results] if results is not None else None

    # --- Bill Table Functions ---

    def get_total_bill_for_month(self, year: int, month: int) -> Optional[float]:
        """
        Calculate the total invoice amount for a given billing period (year and month).

        Args:
            year (int): The year (e.g., 2024).
            month (int): The month (1-12).

        Returns:
            Optional[float]: The total bill amount for the specified period, or None on error.
                             Returns 0.0 if no bills found for the period.
        """
        try:
            # Convert numeric month to full month name (e.g., 8 -> 'August')
            month_name = datetime(year, month, 1).strftime("%B")
            billing_period_str = f"{month_name} {year}"  # Format: 'August 2024'
        except ValueError:
            print(f"Invalid year ({year}) or month ({month}) provided.")
            return None

        query = """
            SELECT SUM(invoice_amount)
            FROM bills
            WHERE billing_period = %s;
        """
        params = (billing_period_str,)
        result = self._execute_query(query, params, fetch_one=True, is_dict_cursor=False)

        # result will be a tuple like (Decimal('total_amount'),) or (None,)
        if result and result[0] is not None:
            return float(result[0]) # Convert Decimal to float
        elif result is not None and result[0] is None:
             return 0.0 # No bills found for this period
        else:
            return None # Error occurred

    def get_bill_details(self, order_id: Optional[int] = None, invoice_number: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve details for a specific bill by order ID or invoice number.

        Args:
            order_id (Optional[int]): The ID of the associated order.
            invoice_number (Optional[str]): The unique invoice number.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing bill details, or None if not found or error.
        """
        if order_id is not None:
            query = "SELECT id, order_id, invoice_number, invoice_amount, bill_paid, billing_period, tax FROM bills WHERE order_id = %s;"
            params = (order_id,)
        elif invoice_number is not None:
            query = "SELECT id, order_id, invoice_number, invoice_amount, bill_paid, billing_period, tax FROM bills WHERE invoice_number = %s;"
            params = (invoice_number,)
        else:
            print("Error: Must provide either order_id or invoice_number.")
            return None

        result = self._execute_query(query, params, fetch_one=True, is_dict_cursor=True)
        # Convert Decimal to float for easier LLM processing if needed
        if result:
            bill_dict = dict(result)
            bill_dict['invoice_amount'] = float(bill_dict['invoice_amount']) if bill_dict.get('invoice_amount') else None
            bill_dict['tax'] = float(bill_dict['tax']) if bill_dict.get('tax') else None
            return bill_dict
        return None

    def get_unpaid_bills(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a list of all bills that are marked as unpaid ('No').

        Returns:
            Optional[List[Dict[str, Any]]]: A list of unpaid bill dictionaries, or None on error.
        """
        query = """
            SELECT b.id, b.order_id, b.invoice_number, b.invoice_amount, b.bill_paid, b.billing_period, b.tax, o.user_id
            FROM bills b
            LEFT JOIN orders o ON b.order_id = o.id -- Join to potentially get user_id
            WHERE b.bill_paid = 'No' -- Or adjust based on your exact value ('false', '0' etc.)
            ORDER BY b.billing_period DESC, b.id;
        """
        results = self._execute_query(query, is_dict_cursor=True)
        if results is not None:
            processed_results = []
            for row in results:
                bill_dict = dict(row)
                bill_dict['invoice_amount'] = float(bill_dict['invoice_amount']) if bill_dict.get('invoice_amount') else None
                bill_dict['tax'] = float(bill_dict['tax']) if bill_dict.get('tax') else None
                processed_results.append(bill_dict)
            return processed_results
        return None

    # --- Combined/Complex Queries ---

    def get_user_total_spending(self, user_id: Optional[int] = None, email: Optional[str] = None) -> Optional[float]:
        """
        Calculate the total amount spent by a specific user on paid bills.

        Args:
            user_id (Optional[int]): The ID of the user.
            email (Optional[str]): The email address of the user.

        Returns:
            Optional[float]: The total amount spent by the user, or None on error. Returns 0.0 if no paid bills found.
        """
        target_user_id = None
        if user_id is not None:
            target_user_id = user_id
        elif email is not None:
            user_details = self.get_user_details(email=email)
            if not user_details:
                print(f"No user found with email: {email}")
                return None
            target_user_id = user_details.get('id') if isinstance(user_details, dict) else None
        else:
            print("Error: Must provide either user_id or email for spending calculation.")
            return None

        query = """
            SELECT SUM(b.invoice_amount)
            FROM bills b
            JOIN orders o ON b.order_id = o.id
            WHERE o.user_id = %s AND b.bill_paid = 'Yes'; -- Assumes 'Yes' means paid
        """
        params = (target_user_id,)
        result = self._execute_query(query, params, fetch_one=True, is_dict_cursor=False)

        if result and result[0] is not None:
            return float(result[0]) # Convert Decimal to float
        elif result is not None and result[0] is None:
            return 0.0 # User exists but has no paid bills
        else:
            return None # Error occurred

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    from config import DB_CONFIG 
    print("Starting DB Interface Example")
    db_interface = DatabaseInterface(DB_CONFIG)

    # --- Test Calls ---
    print("\n--- Testing User Functions ---")
    user_count = db_interface.get_user_count()
    print(f"Total users: {user_count}")

    user_details_by_id = db_interface.get_user_details(user_id=1)
    print(f"User details for ID 1: {user_details_by_id}")

    user_details_by_email = db_interface.get_user_details(email='kelseyshaw@example.net')
    print(f"User details for email kelseyshaw@example.net: {user_details_by_email}")

    users_created = db_interface.get_users_by_creation_date('2024-01-01', '2024-12-31')
    print(f"Users created in 2024: {users_created}")


    print("\n--- Testing Order Functions ---")
    order_status_counts = db_interface.get_order_status_count()
    print(f"Order status counts: {order_status_counts}")

    orders_timeline = db_interface.get_orders_by_timeline('2024-10-01', '2024-10-31')
    print(f"Orders in Oct 2024 summary: {orders_timeline}")

    order_details = db_interface.get_order_details(order_id=1) # Assuming order ID 1 exists
    print(f"Details for Order ID 1: {order_details}")

    orders_for_user = db_interface.get_orders_by_user(user_id=168) # Assuming user ID 168 exists
    print(f"Orders for User ID 168: {orders_for_user}")

    print("\n--- Testing Bill Functions ---")
    total_bill_month = db_interface.get_total_bill_for_month(year=2024, month=8)
    print(f"Total bill amount for August 2024: {total_bill_month}")

    bill_details_order = db_interface.get_bill_details(order_id=1308) # Assuming order ID 1308 exists
    print(f"Bill details for Order ID 1308: {bill_details_order}")

    bill_details_invoice = db_interface.get_bill_details(invoice_number='INV0550') # Assuming INV0550 exists
    print(f"Bill details for Invoice INV0550: {bill_details_invoice}")

    unpaid_bills = db_interface.get_unpaid_bills()
    print(f"Unpaid bills: {unpaid_bills}")

    print("\n--- Testing Combined Functions ---")
    user_spending = db_interface.get_user_total_spending(user_id=1) # Replace with a user ID likely to have paid bills
    print(f"Total spending for User ID 1: {user_spending}")

    print("\nDB Interface Example Finished")