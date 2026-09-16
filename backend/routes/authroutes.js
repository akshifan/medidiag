const express = require("express");
const router = express.Router();
const User = require("../models/user");
const Patient = require("../models/patient");
const Doctor = require("../models/doctor");

// Login page
router.get("/login", (req, res) => {
  res.render("auth/login", {
    title: "Login - MediDiag",
    error: null,
  });
});

// Login processing
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findByEmail(email);
    if (!user) {
      return res.render("auth/login", {
        title: "Login - MediDiag",
        error: "Invalid email or password",
      });
    }

    const isValidPassword = await User.verifyPassword(password, user.password);
    if (!isValidPassword) {
      return res.render("auth/login", {
        title: "Login - MediDiag",
        error: "Invalid email or password",
      });
    }

    req.session.user = {
      id: user.id,
      email: user.email,
      role: user.role,
    };

    // Redirect based on role
    if (user.role === "patient") {
      res.redirect("/patient/dashboard");
    } else if (user.role === "doctor") {
      res.redirect("/doctor/dashboard");
    }
  } catch (error) {
    console.error("Login error:", error);
    res.render("auth/login", {
      title: "Login - MediDiag",
      error: "An error occurred during login",
    });
  }
});

// Registration page
router.get("/register", (req, res) => {
  const role = req.query.role;
  res.render("registerpage/register", {
    title: "Register - MediDiag",
    error: null,
    role,
  });
});

// Registration processing
router.post("/register", async (req, res) => {
  try {
    const {
      email,
      password,
      confirm_password,
      full_name,
      ...additionalData
    } = req.body;

    // Validation
    // if (password !== confirm_password) {
    //   return res.render("registerpage/register", {
    //     title: "Register - MediDiag",
    //     error: "Passwords do not match",
    //   });
    // }

    // if (password.length < 6) {
    //   return res.render("registerpage/register", {
    //     title: "Register - MediDiag",
    //     error: "Password must be at least 6 characters long",
    //   });
    // }

    // Check if user already exists
    const existingUser = await User.findByEmail(email);
    if (existingUser) {
      return res.render("registerpage/register", {
        title: "Register - MediDiag",
        error: "Email already registered",
      });
    }

    // Create user
    // Create user
const role = (req.body.role || req.query.role || "").trim().toLowerCase();

console.log("Registration role:", role);
console.log("Registration body:", req.body);
console.log("Registration query:", req.query);

// Only allow valid application roles
if (!["patient", "doctor"].includes(role)) {
  return res.render("registerpage/register", {
    title: "Register - MediDiag",
    error: "Please select a valid account type.",
    role: role ||null,
  });
}

const user = await User.create(email, password, role);



    // Create role-specific profile
    if (role === "patient") {
      await Patient.create(user.id, {
        full_name,
        date_of_birth: additionalData.date_of_birth,
        gender: additionalData.gender,
        phone: additionalData.phone,
      });
    } else if (role === "doctor") {
      const specialization = additionalData.specialization[0].toUpperCase() + additionalData.specialization.slice(1);
      await Doctor.create(user.id, {
        full_name,
        specialization: specialization,
        license_number: additionalData.license_number,
        experience_years: additionalData.experience_years,
        phone: additionalData.phone,
        hospital_affiliation: additionalData.hospital_affiliation,
        consultation_fee: additionalData.consultation_fee,
      });
    }

    // Auto-login after registration
    req.session.user = {
      id: user.id,
      email: user.email,
      role: user.role,
    };

    // Redirect based on role
    if (role === "patient") {
      res.redirect("/patient/dashboard");
    } else if (role === "doctor") {
      res.redirect("/doctor/dashboard");
    }
  } catch (error) {
    console.error("Registration error:", error);
    res.render("registerpage/register", {
      title: "Register - MediDiag",
      error: "An error occurred during registration",
    });
  }
});

// Logout
router.get("/logout", (req, res) => {
  req.session.destroy();
  res.redirect("/");
});

module.exports = router;