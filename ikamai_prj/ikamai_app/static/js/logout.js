// Open Instruction Modal
function openModal() {
    var modal = document.getElementById("instructionModal");
    modal.style.display = "flex";
}

// Close Instruction Modal
function closeModal() {
    var modal = document.getElementById("instructionModal");
    modal.style.display = "none";
}

// Open Logout Confirmation Modal
function openLogoutModal() {
    var modal = document.getElementById("logoutModal");
    modal.style.display = "flex";
}

// Close Logout Confirmation Modal
function closeLogoutModal() {
    var modal = document.getElementById("logoutModal");
    modal.style.display = "none";
}

// Confirm Logout Action
function confirmLogout() {
    window.location.href = logoutUrl;  // Use the resolved URL
}

// Close Modal when clicking outside
window.onclick = function(event) {
    var instructionModal = document.getElementById("instructionModal");
    var logoutModal = document.getElementById("logoutModal");
    if (event.target === instructionModal) {
        instructionModal.style.display = "none";
    }
    if (event.target === logoutModal) {
        logoutModal.style.display = "none";
    }
};