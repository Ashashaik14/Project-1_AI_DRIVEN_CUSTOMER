function loadChart() {
    const ctx = document.getElementById("customerChart");

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["High Value", "Medium", "Low", "At Risk"],
            datasets: [{
                label: "Customers",
                data: [40, 80, 60, 20],
                backgroundColor: ["#4e73df", "#1cc88a", "#36b9cc", "#e74a3b"]
            }]
        }
    });
}