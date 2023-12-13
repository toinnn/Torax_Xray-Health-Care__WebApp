function handleFileSelect(event) {
    
    // return;
    // Obtém o arquivo selecionado
    const fileInput = event.target;
    const file = fileInput.files[0];

    // alert("me notaaa");
    // console.log("qualqquer coisa so pra testar");

    // Verifica se um arquivo foi selecionado
    if (file) {
        // Cria um objeto FormData para enviar o arquivo
        const formData = new FormData();
        formData.append('file', file);

        // Opções da solicitação
        const options = {
            method: 'POST',
            // method: 'GET',
            body: formData
        };

        // URL do servidor para onde você está enviando a solicitação POST
        const url = '/send_image';

        // Realiza a solicitação POST usando fetch
        // fetch(url, options)
        fetch(url, options)
            .then(response => response.json())
            .then(data => {
                console.log('Resposta do servidor:', data);
                
                // window.location.href = data["nova_url"] ;

                console.log('Resposta do servidor:', data);

                // Converte a imagem para Base64
                const reader = new FileReader();
                reader.onloadend = function () {
                    // Cria uma lista contendo a imagem e outras informações
                    const listaConteudo = [reader.result, data["output"]];

                    // Armazena a lista no localStorage
                    localStorage.setItem('Image_0', JSON.stringify(listaConteudo));

                    exibirNovaImagem()
                    // Redireciona para a nova tela
                    // window.location.href = data["nova_url"];
                };


                reader.readAsDataURL(file);


            })
            .catch(error => {
                console.error('Erro:', error);
            });
        event.preventDefault();
    }
}

// function load_page_scene(){



//     // Obtém a string armazenada no localStorage
//     const listaString = localStorage.getItem('Image_0');

//     // Converte a string de volta para um objeto JavaScript usando JSON.parse
//     const listaConteudo = JSON.parse(listaString);

//     // Agora, listaConteudo contém os dados recuperados do localStorage
//     console.log( "O conteúdo salvo na local storage era : " , listaConteudo);
// }

function load_main_frame(){


    // Obtém a string armazenada no localStorage
    const listaString = localStorage.getItem('Image_0');

    // Converte a string de volta para um objeto JavaScript usando JSON.parse
    const listaConteudo = JSON.parse(listaString);

    // Verifica se a listaConteudo não é nula e tem a imagem na primeira posição
    if (listaConteudo && listaConteudo.length > 0) {
        // Cria um elemento img
        const imagemElemento = document.createElement('img');

        // Define o atributo src com a string Base64 da imagem
        imagemElemento.src = listaConteudo[0];  // Assumindo que a imagem está na primeira posição
        console.log("A resposta do back-end foi : " , novaListaConteudo[1] )
        alterarLabel(novaListaConteudo[1])
        // Adiciona o elemento img ao contêiner de imagem
        document.getElementById('imagem-container').appendChild(imagemElemento);
    } else {
        console.error('Lista de conteúdo inválida.');
    }

}


function exibirNovaImagem() {
    // Obtém a string armazenada no localStorage
    const listaString = localStorage.getItem('Image_0');

    // Converte a string de volta para um objeto JavaScript usando JSON.parse
    const novaListaConteudo = JSON.parse(listaString);
    console.log("A resposta do back-end foi : " , novaListaConteudo[1] )
    // Verifica se a novaListaConteudo não é nula e tem a imagem na primeira posição
    if (novaListaConteudo && novaListaConteudo.length > 0) {
        // Cria um elemento img
        const novaImagemElemento = document.createElement('img');

        // Define o atributo src com a string Base64 da nova imagem
        novaImagemElemento.src = novaListaConteudo[0];  // Assumindo que a nova imagem está na primeira posição
        alterarLabel(novaListaConteudo[1])

        // Obtém o contêiner de imagem existente
        const imagemContainer = document.getElementById('imagem-container');

        // Limpa qualquer conteúdo existente no contêiner
        imagemContainer.innerHTML = '';

        // Adiciona o elemento img ao contêiner de imagem
        imagemContainer.appendChild(novaImagemElemento);
    } else {
        console.error('Nova lista de conteúdo inválida.');
    }
}


// Função para alterar a label
function alterarLabel(labels) {
    // Obtém a nova label (pode ser obtida da sua lógica específica)
    const novaLabel = labels//obterNovaLabel();

    // Obtém o elemento de label existente
    const labelElemento = document.getElementById('nova-label');

    // Atualiza o conteúdo da label
    labelElemento.textContent = `${novaLabel}`;
}

// Função fictícia para obter a nova label (substitua pela sua lógica)
function obterNovaLabel() {
    // Lógica para obter a nova label, substitua conforme necessário
    return 'Nova Label Dinâmica';
}