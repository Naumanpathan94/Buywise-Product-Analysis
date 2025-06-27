// This is a legacy handler for customer reviews (not used in the UI, for reference only).
// Do not include this file in your HTML template.

const customerReviewForm = document.getElementById('customer-review-form');
if (customerReviewForm) {
  customerReviewForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const productName = customerReviewForm.product_name.value;
    const name = customerReviewForm.customer_name.value.trim() || "Anonymous";
    const text = customerReviewForm.customer_review.value.trim();
    const date = new Date().toLocaleString();
    if (!text) return;
    if (!customerReviews[productName]) customerReviews[productName] = [];
    customerReviews[productName].unshift({ name, text, date });
    localStorage.setItem('customerReviews', JSON.stringify(customerReviews));
    renderCustomerReviews(productName);
    customerReviewForm.reset();
  });
}
