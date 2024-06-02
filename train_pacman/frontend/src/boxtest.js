import './styles.css';
import { Application, Graphics} from 'pixi.js';

(async () =>
{
    const app = new Application();

    await app.init({
        canvas: document.getElementById('pixiCanvas'),
        width: 200,
        height: 200
    });

    // document.body.appendChild(app.canvas);

    const box = new Graphics();
    box.rect(0, 0, 50, 50);
    box.fill(0xDE3249)
    app.stage.addChild(box);

    // Fetch initial coordinate
    try {
        const response = await fetch('http://localhost:8000/api/initial-coordinate/');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        box.x = data.x_coordinate;
    } catch (fetchError) {
        console.error('Error fetching initial coordinate:', fetchError);
    }

    // Connect to WebSocket
    const socket = new WebSocket('ws://localhost:8000/ws/box/');

    socket.onopen = () => {
        console.log('WebSocket connection established');
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            box.x = data.x_coordinate;
        } catch (parseError) {
            console.error('Error parsing WebSocket message:', parseError)
        }
    };

})();