import './styles.css';
import { Application, Graphics, Container } from 'pixi.js';
import MazeSprites from './mazesprites';
import Pellets from './pellets';
import PacmanSprites from './pacmansprites';
import GhostSprites from './ghostsprites';
import * as constants from './constants';
import LifeSprites from './lifesprites';
import Rewardbubble from './reward_bubbles';
import ScoreText from './scoretxt';

(async () =>
{
    // function getActualPixels(element) {
    //     const rect = element.getBoundingClientRect();
    //     const width = rect.width * window.devicePixelRatio;
    //     const height = rect.height * window.devicePixelRatio;
    //     return { width, height };
    // }

    const apiBaseUrl = process.env.APP_API_URL || 'train-pacman.com'
    const app = new Application();

    globalThis.__PIXI_APP__ = app;

    const pixiContainer = document.getElementById('pixi-container');
    const pixiCanvas = document.getElementById('pixiCanvas');
    // const { width, height } = pixiContainer.getBoundingClientRect();
    // const { width, height } = getActualPixels(pixiContainer);

    const fixedWidth = constants.SCREENWIDTH;
    const fixedHeight = constants.SCREENHEIGHT;

    const scaleX = 1;
    const scaleY = 1;

    // const scaleX = width / constants.SCREENWIDTH;
    // const scaleY = height / constants.SCREENHEIGHT;

    await app.init({
        canvas: pixiCanvas,
        width: fixedWidth,
        height: fixedHeight,
        antialias: false,
        background: 'black',
    });

    // window.addEventListener('resize', () => resizeCanvas(app));
    // resizeCanvas(app);

    let data;
    try {
        const response = await fetch(`http://${apiBaseUrl}/api/initial-state/`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        data = await response.json();
    } catch (fetchError) {
        console.error('Error fetching initial coordinate:', fetchError);
    }
    let pacman_loc = JSON.parse(data['pacman_loc']);
    let ghost_loc = JSON.parse(data['ghost_loc']);

    const mazefile = await fetch('assets/maze1.txt').then(response => response.text());
    const rotfile = await fetch('assets/maze1_rotation.txt').then(response => response.text());

    const maze = new MazeSprites(mazefile, rotfile, scaleX, scaleY);
    await maze.loadSpritesheet();
    maze.constructBackground(0);
    let background = maze.getBackground();
    app.stage.addChild(background);

    let pellets = new Pellets(scaleX, scaleY);
    app.stage.addChild(pellets.pellets);

    const pacman = new PacmanSprites(scaleX, scaleY);
    await pacman.loadSpritesheet();
    const pacmanSprite = pacman.getSprite();
    app.stage.addChild(pacmanSprite);
    pacman.drawPacman(pacman_loc[0], pacman_loc[1], 0);
    // let sprite = pacman.getSprite();
    // app.stage.addChild(sprite);
    // pacman.drawPacman(0, 0, 0);
    // pacman.changeImage();

    const ghost = new GhostSprites('BLINKY', constants.CHASE, constants.RIGHT, scaleX, scaleY);
    await ghost.loadSpritesheet();
    const ghostSprite = ghost.getSprite();
    app.stage.addChild(ghostSprite);
    ghost.drawGhost(ghost_loc[0], ghost_loc[1], 0);

    const lives = new LifeSprites(scaleX, scaleY);
    await lives.loadSpritesheet();
    lives.resetLives(5);
    const livesContainer = lives.getSprite();
    app.stage.addChild(livesContainer);
    lives.drawLives();

    const scoretxt = new ScoreText(scaleX, scaleY);
    const text = scoretxt.getText();
    app.stage.addChild(text);

    // Connect to WebSocket
    const socket = new WebSocket(`ws://${apiBaseUrl}/ws/pacman/`);

    socket.onopen = () => {
        console.log('WebSocket connection established');
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            const messageType = data['message_type'];
            switch (messageType) {
                case 'state_update':
                    pacman_loc = JSON.parse(data['pacman_loc']);
                    pacman.direction = data['pacman_direction'];
                    ghost_loc = JSON.parse(data['ghost_loc']);
                    ghost.direction = data['ghost_direction'];
                    const pelletList = JSON.parse(data['pellets']);
                    const numLives = JSON.parse(data['num_lives']);
                    const score = JSON.parse(data['score'])
                    // pellets.pelletList = pelletList;
                    pellets.setPelletList(pelletList);
                    pellets.drawPellets();
                    pacman.drawPacman(pacman_loc[0], pacman_loc[1], 1);
                    ghost.drawGhost(ghost_loc[0], ghost_loc[1], 1);
                    lives.resetLives(numLives);
                    lives.drawLives();
                    scoretxt.updateScore(score);
                    break;
                case 'reward_signal':
                    const pos = pacman.getPos()
                    const reward_str = data['reward'];
                    switch (reward_str) {
                        case 'reward_plus1': {
                            let rb = new Rewardbubble('+', pos[0], pos[1], acc, alpha_decay);
                            app.stage.addChild(rb.getGraphic());
                            rewardBubbles.push(rb);
                            break;
                        }
                        case 'reward_plus10': {
                            let rb = new Rewardbubble('++', pos[0], pos[1], acc, alpha_decay);
                            app.stage.addChild(rb.getGraphic());
                            rewardBubbles.push(rb);
                            break;

                        }
                        case 'reward_minus1': {
                            let rb = new Rewardbubble('-', pos[0], pos[1], acc, alpha_decay);
                            app.stage.addChild(rb.getGraphic());
                            rewardBubbles.push(rb);
                            break;
                        }
                        case 'reward_minus10': {
                            let rb = new Rewardbubble('--', pos[0], pos[1], acc, alpha_decay);
                            app.stage.addChild(rb.getGraphic());
                            rewardBubbles.push(rb);
                            break;
                        }

                    }
            }
        } catch (parseError) {
            console.error('Error parsing WebSocket message:', parseError)
        }
    };

    let rewardBubbles = [];
    const acc = 0.5;
    const alpha_decay = 0.05;

    document.getElementById('button_plus1').addEventListener('click', function() {
        this.classList.add('active');
        socket.send(JSON.stringify({ action: 'reward_plus1' }));
        setTimeout(() => this.classList.remove('active'), 200);
        // const pos = pacman.getPos()
        // let rb = new Rewardbubble('+', pos[0], pos[1], acc, alpha_decay);
        // app.stage.addChild(rb.getGraphic());
        // rewardBubbles.push(rb)
    });
    document.getElementById('button_plus10').addEventListener('click', function() {
        this.classList.add('active');
        socket.send(JSON.stringify({ action: 'reward_plus10' }));
        setTimeout(() => this.classList.remove('active'), 200);
        // const pos = pacman.getPos()
        // let rb = new Rewardbubble('++', pos[0], pos[1], acc, alpha_decay);
        // app.stage.addChild(rb.getGraphic());
        // rewardBubbles.push(rb)
    });
    document.getElementById('button_minus1').addEventListener('click', function() {
        this.classList.add('active');
        socket.send(JSON.stringify({ action: 'reward_minus1' }));
        setTimeout(() => this.classList.remove('active'), 200);
        // const pos = pacman.getPos()
        // let rb = new Rewardbubble('-', pos[0], pos[1], acc, alpha_decay);
        // app.stage.addChild(rb.getGraphic());
        // rewardBubbles.push(rb)
    });
    document.getElementById('button_minus10').addEventListener('click', function() {
        this.classList.add('active');
        socket.send(JSON.stringify({ action: 'reward_minus10' }));
        setTimeout(() => this.classList.remove('active'), 200);
        // const pos = pacman.getPos()
        // let rb = new Rewardbubble('--', pos[0], pos[1], acc, alpha_decay);
        // app.stage.addChild(rb.getGraphic());
        // rewardBubbles.push(rb)
    });


    const resize = () => {
        const containerWidth = pixiContainer.clientWidth;
        const containerHeight = pixiContainer.clientHeight;

        const aspectRatio = fixedWidth / fixedHeight;
        let newWidth, newHeight;

        if (containerWidth / containerHeight < aspectRatio) {
            // Container is taller relative to its width
            newWidth = containerWidth;
            newHeight = containerWidth / aspectRatio;
        } else {
            // Container is wider relative to its height
            newWidth = containerHeight * aspectRatio;
            newHeight = containerHeight;
        }

        console.log(aspectRatio);
        console.log(newWidth);
        console.log(newHeight);
        console.log(newWidth / newHeight)

        const scale = newWidth / fixedWidth;

        pixiCanvas.style.width = `${newWidth}px`;
        pixiCanvas.style.height = `${newHeight}px`;
        pixiCanvas.style.transform = `scale(${scale})`;
        pixiCanvas.style.transformOrigin = 'center center';
        pixiCanvas.style.left = `${(containerWidth - newWidth) / 2}px`;
        pixiCanvas.style.top = `${(containerHeight - newHeight) / 2}px`;
        pixiCanvas.style.position = 'absolute';
        
        // const { width, height } = pixiContainer.getBoundingClientRect();
        // // const { width, height } = getActualPixels(pixiContainer);
        // app.renderer.resize(width, height);

        // const scaleX = width / constants.SCREENWIDTH;
        // const scaleY = height / constants.SCREENHEIGHT;

        
        // console.log(width)
        // console.log(height)
        // maze.updateScale(scaleX, scaleY);
        // maze.updateBackground();
        // pacman.updateScale(scaleX, scaleY);
        // pellets.updateScale(scaleX, scaleY);
        // ghost.updateScale(scaleX, scaleY);
        // lives.updateScale(scaleX, scaleY);
        // scoretxt.updateScale(scaleX, scaleY);
    };

    // window.addEventListener('resize', resize);
    // resize();

    app.ticker.add((time) => {
        for (let i = rewardBubbles.length-1; i >= 0; i--) {
            if (rewardBubbles[i].isAlive) {
                rewardBubbles[i].update(time.deltaTime);
            } else {
                app.stage.removeChild(rewardBubbles[i].getGraphic());
                rewardBubbles.splice(i, 1);
            }
        }

    });



})();

