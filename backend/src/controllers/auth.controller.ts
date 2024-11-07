import { Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import prisma from "../utils/prisma";  // Make sure to import the prisma client

const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret';  // Use an environment variable for JWT secret

// Register function - creates a new user
export async function register(req: Request, res: Response): Promise<void> {
    const { email, password, name, mobile } = req.body;

    // Validate input fields (you can use express-validator or Joi here for better validation)
    if (!email || !password || !name || !mobile) {
        res.status(400).json({ message: 'All fields are required' });
        return;
    }

    try {
        // Check if the email already exists
        const existingUser = await prisma.user.findUnique({
            where: { email },
        });
        if (existingUser) {
            res.status(400).json({ message: 'Email is already registered' });
            return;
        }

        // Hash the password before saving it
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create new user in the database
        const user = await prisma.user.create({
            data: {
                email,
                password: hashedPassword,
                name,
                mobile,
            },
        });

        // Respond with the created user (avoid sending password)
        res.status(201).json({
            message: 'User registered successfully',
            user: { id: user.id, email: user.email, name: user.name, mobile: user.mobile },
        });
    } catch (error) {
        console.error('Register failed: ', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}

// Login function - authenticates user
export async function login(req: Request, res: Response): Promise<void> {
    const { email, password } = req.body;

    // Validate input fields
    if (!email || !password) {
        res.status(400).json({ message: 'Email and password are required' });
        return;
    }

    try {
        // Find the user by email
        const user = await prisma.user.findUnique({
            where: { email },
        });
        if (!user) {
            res.status(400).json({ message: 'Invalid email or password' });
            return;
        }

        // Compare the entered password with the stored hashed password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            res.status(400).json({ message: 'Invalid email or password' });
            return;
        }

        // Generate a JWT token
        const token = jwt.sign(
            { id: user.id, email: user.email },
            JWT_SECRET,
            { expiresIn: '1h' }  // Token expires in 1 hour
        );

        // Respond with the token and user data (avoid sending the password)
        res.status(200).json({
            message: 'Login successful',
            token,
            user: { id: user.id, email: user.email, name: user.name, mobile: user.mobile },
        });
    } catch (error) {
        console.error('Login failed: ', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}

module.exports = {login , register}