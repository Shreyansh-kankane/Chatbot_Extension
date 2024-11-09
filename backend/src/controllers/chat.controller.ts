import axios from 'axios';
import { Request, Response } from 'express';

// Define the expected structure of the request body
interface QueryRequestBody {
    domain: string;
    question: string;
}

export const queryHandler =  async(req: Request, res: Response): Promise<void> =>{
    try {
        const { domain, question } = req.body as QueryRequestBody;

        // Validate input
        if (!domain || !question) {
            res.status(400).json({ message: 'Domain and question are required.' });
            return ;
        }

        // Define the data payload for the AI Engine request
        const data = {
            domain: domain,
            query: { question: question }
        };

        // Send request to AI Engine's /query endpoint
        const response = await axios.post('http://localhost:8000/query', data);

        // Return the response from the AI Engine to the client
        res.status(response.status).json(response.data);
        return ;
    } catch (error) {
        console.error('Error querying AI Engine:', (error as Error).message);
        
        // Send error response to client
        res.status(500).json({ message: 'An error occurred while processing the query.' });
        return ;
    }
}
