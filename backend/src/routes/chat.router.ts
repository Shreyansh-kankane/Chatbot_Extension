import { Router } from 'express';
import {queryHandler} from '../controllers/chat.controller';

const chatRouter = Router();

// Correct usage of the imported controller functions
chatRouter.post('/query', queryHandler);



export default chatRouter;
