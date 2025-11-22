function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 20;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        
        // Random size
        const size = Math.random() * 5 + 3;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        
        // Random position
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        particle.style.left = `${posX}%`;
        particle.style.top = `${posY}%`;
        
        // Random animation duration and delay
        const duration = Math.random() * 15 + 10;
        const delay = Math.random() * 5;
        particle.style.animationDuration = `${duration}s`;
        particle.style.animationDelay = `${delay}s`;
        
        particlesContainer.appendChild(particle);
    }
}

// Input focus effects
const formGroups = document.querySelectorAll('.form-group');
formGroups.forEach(group => {
    const input = group.querySelector('input');
    
    input.addEventListener('focus', () => {
        group.classList.add('focus');
    });
    
    input.addEventListener('blur', () => {
        group.classList.remove('focus');
    });
});

// Password toggle visibility
const togglePassword = document.getElementById('togglePassword');
const passwordInput = document.getElementById('password');

togglePassword.addEventListener('click', () => {
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        togglePassword.classList.remove('fa-eye');
        togglePassword.classList.add('fa-eye-slash');
    } else {
        passwordInput.type = 'password';
        togglePassword.classList.remove('fa-eye-slash');
        togglePassword.classList.add('fa-eye');
    }
});

    // Dark mode toggle
        const darkModeToggle = document.getElementById('darkModeToggle');
        const body = document.body;
        const darkModeIcon = darkModeToggle.querySelector('i');
        
        darkModeToggle.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            
            if (body.classList.contains('dark-mode')) {
                darkModeIcon.classList.remove('fa-moon');
                darkModeIcon.classList.add('fa-sun');
            } else {
                darkModeIcon.classList.remove('fa-sun');
                darkModeIcon.classList.add('fa-moon');
            }
        });

        // Logo hover effect
        const logo = document.getElementById('logo');
        if (logo) {
            logo.addEventListener('mousemove', (e) => {
                const boundingRect = logo.getBoundingClientRect();
                const mouseX = e.clientX - boundingRect.left;
                const mouseY = e.clientY - boundingRect.top;
                
                const centerX = boundingRect.width / 2;
                const centerY = boundingRect.height / 2;
                
                const moveX = (mouseX - centerX) / 20;
                const moveY = (mouseY - centerY) / 20;
                
                logo.style.transform = `translate(${moveX}px, ${moveY}px) scale(1.05)`;
            });
            
            logo.addEventListener('mouseleave', () => {
                logo.style.transform = 'translate(0, 0) scale(1)';
            });
        }

        // Initialize animations
        document.addEventListener('DOMContentLoaded', () => {
            createParticles();
        });

// Form validation and submission
const loginForm = document.getElementById('loginForm');
const usernameGroup = document.getElementById('usernameGroup');
const passwordGroup = document.getElementById('passwordGroup');
const usernameError = document.getElementById('username-error');
const passwordError = document.getElementById('password-error');
const loginButton = document.getElementById('loginButton');
const loginLoader = document.getElementById('loginLoader');
const successCheck = document.getElementById('successCheck');
const usernameInput = document.getElementById('username');
const buttonText = loginButton.querySelector('span');
const buttonIcon = loginButton.querySelector('.fa-sign-in-alt');

loginForm.addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent default form submission
    
    let isValid = true;
    
    // Reset error states
    usernameGroup.classList.remove('error');
    passwordGroup.classList.remove('error');
    usernameError.style.display = 'none';
    passwordError.style.display = 'none';
    
    // Remove any previous error messages
    const previousErrors = document.querySelectorAll('.messages');
    previousErrors.forEach(error => error.remove());
    
    if (!usernameInput.value.trim()) {
        usernameGroup.classList.add('error');
        usernameError.style.display = 'block';
        isValid = false;
    }
    
    if (!passwordInput.value.trim()) {
        passwordGroup.classList.add('error');
        passwordError.style.display = 'block';
        isValid = false;
    }
    
    if (!isValid) {
        return;
    }
    
    // Show loading state
    buttonText.textContent = 'Logging in';
    buttonIcon.style.display = 'none';
    loginLoader.style.display = 'block';
    loginButton.disabled = true;
    
    // Get form data
    const formData = new FormData(loginForm);
    
    // Submit form using fetch API
    fetch(loginForm.action, {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        },
        redirect: 'follow' // Important: follow redirects
    })
    .then(response => {
        if (response.redirected) {
            // Success - redirect to the URL the server is pointing us to
            window.location.href = response.url;
            return null;
        } else if (response.ok) {
            // Got a response but not redirected - this means authentication failed
            return response.text();
        } else {
            throw new Error('Server error: ' + response.status);
        }
    })
    .then(html => {
        if (html) {
            // Reset button state
            loginLoader.style.display = 'none';
            buttonIcon.style.display = 'inline-block';
            buttonText.textContent = 'Login';
            loginButton.disabled = false;
            
            // Create an error message div
            const messagesDiv = document.createElement('div');
            messagesDiv.className = 'messages';
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert alert-error';
            alertDiv.style = ' border-radius: 8px; font-weight: bold; background-color: #ffcccc; color: #d9534f; border: 1px solid #d9534f;';
            alertDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i> Invalid username or password. Please try again.';
            
            messagesDiv.appendChild(alertDiv);
            
            // Find where to insert the error message
            const leftDiv = document.querySelector('.left');
            
            // Insert before the form
            leftDiv.insertBefore(messagesDiv, loginForm);
            
            // Set a timeout to remove the error message after 3 seconds
            setTimeout(() => {
                // Add fade-out class to trigger the CSS transition
                messagesDiv.classList.add('fade-out');
                
                // Remove from DOM after fade completes
                messagesDiv.addEventListener('transitionend', () => {
                    messagesDiv.remove();
                }, { once: true }); // Ensure the event listener is removed after execution
            }, 3000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        loginLoader.style.display = 'none';
        buttonIcon.style.display = 'inline-block';
        buttonText.textContent = 'Login';
        loginButton.disabled = false;
        
        // Create and display a generic error message
        const messagesDiv = document.createElement('div');
        messagesDiv.className = 'messages';
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-error';
        alertDiv.style = 'border-radius: 8px; font-weight: bold; background-color: #ffcccc; color: #d9534f; border: 1px solid #d9534f;';
        alertDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i> An error occurred. Please try again later.';
        
        messagesDiv.appendChild(alertDiv);
        
        const leftDiv = document.querySelector('.left');
        leftDiv.insertBefore(messagesDiv, loginForm);
        
        // Set a timeout to remove the error message after 3 seconds
        setTimeout(() => {
            // Add fade-out class to trigger the CSS transition
            messagesDiv.classList.add('fade-out');
            
            // Remove from DOM after fade completes
            messagesDiv.addEventListener('transitionend', () => {
                messagesDiv.remove();
            }, { once: true }); // Ensure the event listener is removed after execution
        }, 3000);
    });

    
});