import { Request, Response } from 'express';

export async function login(req: Request, res: Response): Promise<void> {
    try {
        res.json("login success");
    } catch (error) {
        console.error("Login failed: ", error);
        res.status(500).json({ message: "Internal server error" });
    }
}


export async function register(req: Request, res: Response): Promise<void> {
    try {
        res.json("register success");
    } catch (error) {
        console.error("register failed: ", error);
        res.status(500).json({ message: "Internal server error" });
    }
}

module.exports = { login , register};